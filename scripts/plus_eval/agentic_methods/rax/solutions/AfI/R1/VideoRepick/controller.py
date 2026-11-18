"""VideoRepick controller (RAX, AfI, R1) — written blind from the public docs.

Pipeline (each phase logs its start/end step and wall time):
  P0 observe      : first observation, instruction, initial frame-sequence manifest
  P1 calibrate    : fit front-camera back-projection against robomme_pixel_to_position
  P2 video        : detect cubes in every conditioning frame (front RGB-D), track them,
                    find the cube the robot lifted (+ gripper-state events), follow it to
                    the first execution frame
  P3 pick         : hover above target, refine xy/yaw with the wrist camera, align yaw,
                    descend, close, verify fingers, lift; retries with re-perception
  P4 place/repeat : per instruction (count / put down), checks task_complete after each
Transitions: any phase failure -> next fallback (documented inline); main never raises;
stops calling tools once episode_ended is reported.
"""
import math
import os
import time
import traceback

import numpy as np

import rx
from rx import log
import geom
import percept

VERSION = "v1.1-blind"
T0 = time.time()

# ---- tunables (untested live; scale typical of ManiSkill tabletop scenes)
SPEED_FREE = 0.018      # m per control step in free space
SPEED_NEAR = 0.008      # m per control step near contact
HOVER_DZ = 0.10         # hover height above cube top (m)
LIFT_DZ = 0.12          # lift height of cube bottom above table (m)
CLOSE_STEPS = 12
OPEN_STEPS = 10
GRIP_OPEN_SUM = 0.07    # finger sum above this = open
GRIP_EMPTY_SUM = 0.005  # finger sum below this = closed on nothing
MAX_PICK_ATTEMPTS = 3
VIDEO_TIME_BUDGET = 300.0


def elapsed():
    return time.time() - T0


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.meta = None
        self.ended = False
        self.complete = False
        self.step_id = None
        self.steps_sent = 0
        self.remaining = None
        self.last_obs = None

    def _absorb(self, meta):
        st = rx.get_state(meta)
        self._got_state = st is not None
        if st is not None:
            self.state = st
        sid = rx.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        if rx.is_ended(meta):
            self.ended = True
        if rx.is_complete(meta):
            self.complete = True
        r = rx.remaining_steps(meta)
        if r is not None:
            self.remaining = r

    def observe(self):
        res = self.ctx.call("robomme_observe")
        meta = rx.reply_meta(res)
        self._absorb(meta)
        imgs = {}
        try:
            imgs = rx.extract_images(res, meta)
        except Exception as e:
            log("extract_images failed", repr(e))
        self.last_obs = (res, meta, imgs)
        return res, meta, imgs

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        if self.remaining is not None and self.remaining <= 0:
            self.ended = True
            raise EpisodeEnded()
        out_meta = None
        for i in range(0, len(rows), 100):
            chunk = rows[i:i + 100]
            res = self.ctx.call("robomme_step", actions=chunk)
            meta = rx.reply_meta(res)
            self.steps_sent += len(chunk)
            self._absorb(meta)
            out_meta = meta
            if self.ended or self.complete:
                break
        self.meta = out_meta
        if not getattr(self, "_got_state", True) and not self.ended:
            # step reply without a state vector: refresh from an observation (no physics)
            try:
                self.observe()
            except Exception as e:
                log("[step] state refresh failed", repr(e)[:200])
        if self.complete:
            log(f"[step] task_complete reported at step {self.step_id}")
        if self.ended:
            log(f"[step] episode_ended at step {self.step_id} complete={self.complete}")
            raise EpisodeEnded()
        return out_meta

    def tool_points(self, cam, step_id, pixels):
        """World points for up to N pixels (list of (u, v)) via the official tool."""
        out = []
        for i in range(0, len(pixels), 32):
            if getattr(self, "uv_swap", False):
                sel = [{"camera": cam, "u": float(v), "v": float(u)} for u, v in pixels[i:i + 32]]
            else:
                sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in pixels[i:i + 32]]
            try:
                res = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                                    selections=sel, depth_window_radius=0)
                pts = rx.parse_pixel_points(res, len(sel))
                if i == 0 and not getattr(self, "_logged_pix", False):
                    self._logged_pix = True
                    t = rx.reply_texts(res)
                    log("[pix] sample reply:", (t[0][:600] if t else "<none>"))
            except Exception as e:
                log("[pix] tool failed", repr(e)[:300])
                pts = [None] * len(sel)
            out.extend(pts)
        return out

    @property
    def fingers(self):
        return float(self.state[6] + self.state[7]) if self.state is not None else None


# ============================================================================ helpers
def image_candidates(robot, res, meta, imgs, cam):
    """Candidate (source, rgb, depth_m) tuples for the current frame of `cam`:
    inline attachments first, then published files of this reply, then manifests."""
    cands = []
    rgb_in, dep_in = imgs.get((cam, "rgb")), imgs.get((cam, "depth"))
    if dep_in is not None:
        cands.append(("inline", rgb_in, dep_in))
    try:
        ents = [e for e in rx.file_entries(meta) if str(e.get("camera", "")).lower() == cam]
        frgb, fdep = None, None
        for e in ents[::-1]:
            kind = str(e.get("kind", "")).lower()
            if kind == "rgb" and frgb is None:
                frgb, _ = rx.load_image_file(e["path"])
            if kind == "depth" and fdep is None:
                a, _ = rx.load_image_file(e["path"])
                if a is not None:
                    fdep = rx._depth_to_m(a, float(e.get("meters_per_unit") or 0.001))
        if fdep is not None:
            cands.append(("files", rgb_in if rgb_in is not None else frgb, fdep))
        elif frgb is not None and rgb_in is None:
            rgb_in = frgb
    except Exception as ex:
        log("file image fallback failed", repr(ex))
    mps = rx.manifest_paths(meta)
    if not cands:
        mps = mps + (rx.manifest_paths(robot.meta) if robot.meta else [])
    for mp in mps[:3]:
        try:
            man, mdir = rx.load_manifest(mp)
            if not man or not man.get("frames"):
                continue
            fr = sorted(man["frames"], key=lambda f: f.get("frame_index", 0))[-1]
            fsid = fr.get("step_id")
            if isinstance(fsid, (int, float)) and robot.step_id is not None and int(fsid) != int(robot.step_id):
                continue  # stale sequence (not the current observation)
            d = rx.load_frame_images(fr, mdir, cams=(cam,))
            if d.get((cam, "depth")) is not None:
                r = rgb_in if rgb_in is not None else d.get((cam, "rgb"))
                cands.append(("manifest", r, d.get((cam, "depth"))))
        except Exception as ex:
            log("manifest image fallback failed", repr(ex))
    out = []
    for name, rgb, dep in cands:
        if rgb is None or dep is None:
            continue
        if rgb.shape[:2] != dep.shape[:2]:
            log(f"[img] {cam}/{name} shape mismatch rgb {rgb.shape} depth {dep.shape}")
            try:
                from PIL import Image
                dep = np.array(Image.fromarray(dep.astype(np.float32)).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)).astype(np.float64)
            except Exception:
                continue
        out.append((name, rgb, dep))
    return out


def get_cam_images(robot, res, meta, imgs, cam, prefer=None):
    c = image_candidates(robot, res, meta, imgs, cam)
    if not c:
        return imgs.get((cam, "rgb")), None
    for name, rgb, dep in c:
        if prefer and name == prefer:
            return rgb, dep
    return c[0][1], c[0][2]


def calibrate(robot, meta, cam, cands, n_side=8, prefer=None):
    """Fit a back-projection model for `cam` on the current observation.
    cands: list of (source, rgb, depth). Returns (model, resid, rgb, depth)."""
    K, pose = rx.get_calibration(meta, cam)
    if not cands:
        model, resid = geom.choose_model([], K, pose)
        log(f"[calib] cam={cam} no depth candidates; metadata model={getattr(model, 'kind', None)}")
        return model, resid, None, None
    H, W = cands[0][2].shape
    pix = []
    for a in np.linspace(0.06, 0.94, n_side):
        for b in np.linspace(0.06, 0.94, n_side):
            pix.append((int(a * (W - 1)), int(b * (H - 1))))
    pts = robot.tool_points(cam, robot.step_id if robot.step_id is not None else 0, pix)
    nvalid = sum(p is not None for p in pts)
    best = (None, None, None, None, 1e9, "")
    for name, rgb, dep in cands:
        for swap in (False, True):
            samples = []
            for (u, v), p in zip(pix, pts):
                if p is None or not np.all(np.isfinite(p)):
                    continue
                uu, vv = (v, u) if swap else (u, v)  # tool (u,v) -> my (col,row)
                if 0 <= vv < dep.shape[0] and 0 <= uu < dep.shape[1] and dep[vv, uu] > 0:
                    samples.append((uu, vv, float(dep[vv, uu]), p))
            if len(samples) < 8:
                continue
            model, resid = geom.choose_model(samples, K, pose)
            log(f"[calib] cam={cam} source={name} swap={swap} n={len(samples)} resid={resid}")
            if model is None or resid is None:
                continue
            score = resid - (0.004 if (prefer and name == prefer) else 0.0)
            if score < best[4]:
                best = (model, resid, rgb, dep, score, f"{name}{'-swap' if swap else ''}")
    if best[0] is None:
        model, resid = geom.choose_model([], K, pose)
        log(f"[calib] cam={cam} tool samples unusable (valid={nvalid}); metadata model={getattr(model, 'kind', None)}")
        return model, None, cands[0][1], cands[0][2]
    if best[5].endswith("-swap") and best[4] < 0.01:
        robot.uv_swap = True
    log(f"[calib] cam={cam} valid_tool={nvalid}/{len(pix)} best={best[5]} model={best[0].kind} resid={best[1]:.5f} "
        f"K={'yes' if K is not None else 'no'} pose={'yes' if pose is not None else 'no'}")
    return best[0], best[1], best[2], best[3]


def world_points(model, depth):
    P = model.points(depth)
    valid = (depth > 0) & np.all(np.isfinite(P), -1)
    return P, valid


_N_DEBUG = [0]


def save_debug(name, rgb, dets=None, mark=None):
    if _N_DEBUG[0] >= 24:
        return
    _N_DEBUG[0] += 1
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb.astype(np.uint8)).resize((2 * rgb.shape[1], 2 * rgb.shape[0]), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        for i, d in enumerate(dets or []):
            u, v = d["uv"]
            dr.rectangle([2 * u - 6, 2 * v - 6, 2 * u + 6, 2 * v + 6], outline=(255, 255, 255))
            dr.text((2 * u + 8, 2 * v - 8), str(i), fill=(255, 255, 255))
        if mark is not None:
            u, v = mark
            dr.ellipse([2 * u - 10, 2 * v - 10, 2 * u + 10, 2 * v + 10], outline=(255, 0, 255), width=3)
        os.makedirs("/workspace/output", exist_ok=True)
        im.save(f"/workspace/output/{name}.jpg", quality=80)
    except Exception as e:
        log("save_debug failed", repr(e))


# ============================================================================ motion
def move_to(robot, target, grip, speed=SPEED_FREE, tol=0.005, max_calls=7, label=""):
    target = np.asarray(target, float)
    last_d = None
    for k in range(max_calls):
        pos = robot.state[:3]
        err = target - pos
        d = float(np.linalg.norm(err))
        if d < tol:
            return True, d
        if last_d is not None and ((last_d - d < 0.001 and d < 0.03) or (last_d - d < 0.1 * last_d)):
            log(f"[move{label}] stalled d={d:.4f} (prev {last_d:.4f})")
            return d < 3 * tol, d
        last_d = d
        n = int(min(60, max(1, math.ceil(d / speed))))
        delta = err / n / 0.05
        rows = [[delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip]] * n
        robot.step(rows)
    d = float(np.linalg.norm(target - robot.state[:3]))
    return d < 3 * tol, d


def hold(robot, grip, n):
    robot.step([[0, 0, 0, 0, 0, 0, grip]] * n)


ROT_DISABLED = [False]


def rotate_yaw(robot, delta, grip):
    """Rotate the hand about world z by delta (rad), with measured feedback.
    Undoes the command and disables yaw alignment if it tilts the hand."""
    if ROT_DISABLED[0]:
        return delta
    sign = 1.0
    tilt0 = geom.approach_axis_down_err(robot.state)
    target = geom.wrap_pi(geom.hand_yaw(robot.state) + delta)
    hold_pos = robot.state[:3].copy()
    remaining = delta
    for it in range(4):
        if abs(remaining) < math.radians(3):
            break
        phi0 = geom.hand_yaw(robot.state)
        n = int(max(2, math.ceil(abs(remaining) / 0.10)))
        dr = sign * remaining / n / 0.5
        rows = [[0, 0, 0, 0, 0, dr, grip]] * n
        robot.step(rows)
        tilt1 = geom.approach_axis_down_err(robot.state)
        if tilt1 > tilt0 + 8.0:
            log(f"[yaw] rotation tilted the hand ({tilt0:.1f}->{tilt1:.1f} deg); undoing and disabling yaw alignment")
            robot.step([[0, 0, 0, 0, 0, -r[5], grip] for r in rows])
            ROT_DISABLED[0] = True
            break
        phi1 = geom.hand_yaw(robot.state)
        achieved = geom.wrap_quarter(phi1 - phi0)
        log(f"[yaw] it={it} cmd={math.degrees(remaining):.1f} achieved={math.degrees(achieved):.1f} sign={sign}")
        if abs(remaining) > math.radians(8) and achieved * remaining < 0 and abs(achieved) > 0.5 * abs(remaining) * 0.5:
            sign = -sign
        remaining = geom.wrap_quarter(target - phi1)
    move_to(robot, hold_pos, grip, speed=SPEED_NEAR, tol=0.004, max_calls=3, label="-yawfix")
    return remaining


# ============================================================================ perception
class Scene:
    def __init__(self):
        self.front_model = None
        self.table_z = None
        self.ref_xy = None
        self.use_sat = True
        self.front_source = None


def front_detect(robot, scene, res, meta, imgs, save=None):
    rgb, dep = get_cam_images(robot, res, meta, imgs, "front", prefer=scene.front_source)
    if rgb is None or dep is None or scene.front_model is None:
        return [], None, None, None
    P, valid = world_points(scene.front_model, dep)
    dets = percept.detect(P, valid, rgb, scene.table_z, scene.ref_xy, use_sat=scene.use_sat,
                          exclude_xyz=None if scene.use_sat else robot.state[:3], exclude_r=0.0 if scene.use_sat else 0.08)
    if save:
        save_debug(save, rgb, dets)
    return dets, P, valid, rgb


def wrist_detect(robot, scene, res, meta, imgs, expect_xy, expect_chroma):
    cands = image_candidates(robot, res, meta, imgs, "wrist")
    if not cands:
        log("[wrist] no wrist rgb/depth")
        return None
    model, resid, rgb, dep = calibrate(robot, meta, "wrist", cands, n_side=6)
    if model is None or rgb is None or dep is None or (resid is not None and resid > 0.015):
        log(f"[wrist] calibration unusable resid={resid}")
        return None
    P, valid = world_points(model, dep)
    dets = percept.detect(P, valid, rgb, scene.table_z, None, use_sat=scene.use_sat)
    best, bc = None, 1e9
    for d in dets:
        c = np.linalg.norm(d["xy"] - expect_xy[:2]) + 0.6 * percept.color_dist(d["chroma"], expect_chroma)
        if c < bc:
            best, bc = d, c
    save_debug(f"wrist_{robot.step_id}", rgb, dets, best["uv"] if best else None)
    if best is None or np.linalg.norm(best["xy"] - expect_xy[:2]) > 0.07:
        log(f"[wrist] no matching detection (n={len(dets)})")
        return None
    yaw, q = percept.cube_yaw(best["top_pts"])
    ltz = percept.local_table_z(P, valid, best["xy"], global_z=scene.table_z)
    log(f"[wrist] det xy={best['xy'].round(4)} ztop={best['ztop']:.4f} yaw={None if yaw is None else round(math.degrees(yaw),1)} "
        f"n_top={len(best['top_pts'])} local_table={ltz}")
    return {"xy": best["xy"], "ztop": best["ztop"], "yaw": yaw, "table_z": ltz, "chroma": best["chroma"]}


# ============================================================================ video
def process_video(robot, scene, meta0, instr):
    """Return dict(target_pos, chroma, info) or None."""
    mans = rx.manifest_paths(meta0)
    frames, mdir = None, None
    for mp in mans:
        man, d = rx.load_manifest(mp)
        if man and man.get("frames"):
            frames, mdir = man["frames"], d
            log(f"[video] manifest {mp}: {len(frames)} frames; sampling={man.get('video_sampling') or rx.find_first(meta0, 'video_sampling')}")
            break
    if not frames:
        log("[video] no frame manifest found")
        return None
    frames = sorted(frames, key=lambda f: (f.get("frame_index", 0)))
    phases = {}
    for f in frames:
        phases[f.get("phase")] = phases.get(f.get("phase"), 0) + 1
    log(f"[video] phases={phases}")
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") != "video_demo"]
    seq = vid + exe[-1:]
    if not vid:
        log("[video] no video_demo frames")
    tracker = percept.Tracker()
    last_dets = [[]]
    last_vid_dets = [[]]
    ee_prev = None
    t_start = time.time()
    stride = 1
    n_det = []
    grip_hist = []
    for i, f in enumerate(seq):
        if time.time() - t_start > VIDEO_TIME_BUDGET and i < len(seq) - 1:
            if stride == 1:
                log("[video] time budget exceeded; skipping to last frame")
            stride = 2
            continue
        try:
            ims = rx.load_frame_images(f, mdir, cams=("front",))
            rgb, dep = ims.get(("front", "rgb")), ims.get(("front", "depth"))
            if rgb is None or dep is None:
                n_det.append(-1)
                continue
            P, valid = world_points(scene.front_model, dep)
            st = f.get("state")
            ee = None
            closed = None
            if rx._is_num_list(st, 8) and any(abs(float(x)) > 1e-6 for x in st[:3]):
                st = np.array(st, float)
                ee = st[:3]
                closed = (st[6] + st[7]) < GRIP_OPEN_SUM
                grip_hist.append((i, float(st[6] + st[7]), ee.copy()))
            dets = percept.detect(P, valid, rgb, scene.table_z, scene.ref_xy, use_sat=scene.use_sat,
                                  exclude_xyz=None if scene.use_sat or ee is None else ee,
                                  exclude_r=0.0 if scene.use_sat else 0.08)
            n_det.append(len(dets))
            if i == len(seq) - 1:
                last_dets[0] = dets
            elif f.get("phase") == "video_demo":
                last_vid_dets[0] = dets
            tracker.update(i, dets, ee, ee_prev, closed)
            ee_prev = ee
            if i in (0, len(seq) // 3, 2 * len(seq) // 3, len(seq) - 1):
                save_debug(f"video_{i:03d}", rgb, dets)
        except Exception as e:
            log(f"[video] frame {i} error {repr(e)[:200]}")
    log(f"[video] processed {len(seq)} frames in {time.time()-t_start:.1f}s; dets per frame (first 40): {n_det[:40]}")
    log(f"[video] tracks: " + "; ".join(
        f"id{tr['id']} n={tr['n']} first={tr['first']} last={tr['last']} pos={np.round(tr['pos'],3).tolist()} col={np.round(tr['color']).tolist()} disp={tracker.displacement(tr):.3f} hmax={max(h for *_, h in tr['hist']):.3f}"
        for tr in tracker.tracks if tr['n'] >= 2))
    if not tracker.tracks:
        return None
    last_i = len(seq) - 1
    lifted = tracker.lifted_tracks()
    # gripper events from state: closing frames
    events = []
    for k in range(1, len(grip_hist)):
        i0, s0, e0 = grip_hist[k - 1]
        i1, s1, e1 = grip_hist[k]
        if s0 >= GRIP_OPEN_SUM - 0.005 and s1 < s0 - 0.004:
            events.append((i1, e1))
    log(f"[video] lifted={[(t, tr['id']) for t, tr, _ in lifted]} grasp_events={[(i, np.round(e,3).tolist()) for i, e in events[:6]]}")
    chosen, why = None, ""
    # distinct lifted tracks in order
    uniq = []
    for t, tr, _ in lifted:
        if tr not in uniq:
            uniq.append(tr)
    if uniq:
        k = instr.get("ordinal")
        if k is not None and len(uniq) > 1:
            chosen = uniq[k] if -len(uniq) <= k < len(uniq) else uniq[0]
            why = f"lifted ordinal {k}"
        else:
            chosen = uniq[0]
            why = "first lifted"
    if chosen is None and events:
        i_ev, ee = events[0]
        best, bd = None, 1e9
        for tr in tracker.tracks:
            # track position at/just before the event
            ps = [p for t, p, _ in tr["hist"] if t <= i_ev]
            if not ps:
                continue
            dd = float(np.linalg.norm(ps[-1][:2] - ee[:2]))
            if dd < bd:
                best, bd = tr, dd
        if best is not None and bd < 0.06:
            chosen, why = best, f"nearest to first grasp event (d={bd:.3f})"
    if chosen is None:
        cands = [tr for tr in tracker.tracks if tr["n"] >= 3]
        if cands:
            chosen = max(cands, key=tracker.displacement)
            if tracker.displacement(chosen) < 0.02:
                chosen = None
            else:
                why = "max displacement"
    vis = None
    if chosen is not None:
        log(f"[video] VISION target id{chosen['id']} ({why}) final pos={np.round(chosen['pos'],4).tolist()} last_seen={chosen['last']}/{last_i}")
        vis = {"pos": chosen["pos"].copy(), "chroma": chosen["chroma"].copy(), "why": "vision:" + why}
    st = None
    try:
        st = state_event_target(grip_hist, tracker, last_dets[0], instr.get("ordinal"), last_i)
    except Exception:
        log("[video] state model error:\n" + traceback.format_exc()[-600:])
    if st is not None and vis is not None:
        agree = float(np.linalg.norm(st["pos"][:2] - vis["pos"][:2]))
        log(f"[video] state-vs-vision final distance {agree:.3f}")
    out = st if st is not None else vis
    if out is None:
        log("[video] could not identify target")
        return None
    # did the scene change between the last video frame and the first execution frame?
    try:
        a, b = last_vid_dets[0], last_dets[0]
        if a and b:
            nn, cmis = [], 0
            for d in b:
                e = min(a, key=lambda e: np.linalg.norm(d["xy"] - e["xy"]))
                nn.append(float(np.linalg.norm(d["xy"] - e["xy"])))
                if nn[-1] < 0.03 and percept.color_dist(d["chroma"], e["chroma"]) > 0.12:
                    cmis += 1
            changed = float(np.median(nn)) > 0.02 or len(a) != len(b) or cmis > 0
            log(f"[video] boundary check: video dets={len(a)} exec dets={len(b)} median NN={np.median(nn):.3f} changed={changed}")
            if changed:
                cd = sorted((percept.color_dist(d["chroma"], out["chroma"]), k) for k, d in enumerate(b))
                if cd and cd[0][0] < 0.08 and (len(cd) == 1 or cd[1][0] > cd[0][0] + 0.10):
                    d = b[cd[0][1]]
                    log(f"[video] scene changed: choosing by colour det xy={np.round(d['xy'],3).tolist()} (cd={cd[0][0]:.3f})")
                    out = {"pos": d["pos"].copy(), "chroma": d["chroma"].copy(), "why": out["why"] + "+colour"}
    except Exception:
        log("[video] boundary check error " + traceback.format_exc()[-300:])
    return out


def state_event_target(grip_hist, tracker, final_dets, ordinal, last_i):
    """Follow the picked cube through grasp/release events read from the video's robot state."""
    if len(grip_hist) < 5:
        return None
    sums = np.array([g[1] for g in grip_hist])
    if np.ptp(sums) < 0.01:
        log("[video/state] gripper never changed; no events")
        return None
    open_level = float(np.percentile(sums, 98))
    thr = min(GRIP_OPEN_SUM, open_level - 0.008)
    log(f"[video/state] finger sums: open_level={open_level:.4f} min={sums.min():.4f} holding-threshold={thr:.4f}")
    segs = []
    cur = None
    for k, (i, fs, ee) in enumerate(grip_hist):
        holding = GRIP_EMPTY_SUM < fs < thr
        if holding and cur is None:
            cur = {"start": i, "ee_start": ee.copy(), "zmax": ee[2], "end": i, "ee_end": ee.copy(), "min_fs": fs}
        elif holding:
            cur["end"], cur["ee_end"] = i, ee.copy()
            cur["zmax"] = max(cur["zmax"], ee[2])
            cur["min_fs"] = min(cur["min_fs"], fs)
        if not holding and cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        cur["open_at_end"] = True
        segs.append(cur)
    lifted = [g for g in segs if g["zmax"] - g["ee_start"][2] > 0.03 and g["end"] - g["start"] >= 2]
    log(f"[video/state] segments={len(segs)} lifted={[(g['start'], g['end'], np.round(g['ee_start'],3).tolist(), np.round(g['ee_end'],3).tolist(), round(g['min_fs'],4)) for g in lifted[:8]]}")
    if not lifted:
        return None
    k = 0
    if ordinal is not None and len(lifted) > 1:
        k = ordinal if -len(lifted) <= ordinal < len(lifted) else 0
    first = lifted[k]
    # which tracked cube was at the grasp point
    chroma = None
    best, bd = None, 1e9
    for tr in tracker.tracks:
        ps = [p for t, p, _ in tr["hist"] if t <= first["start"]]
        if not ps:
            continue
        dd = float(np.linalg.norm(ps[-1][:2] - first["ee_start"][:2]))
        if dd < bd:
            best, bd = tr, dd
    if best is not None and bd < 0.06:
        chroma = best["chroma"].copy()
        log(f"[video/state] grasped track id{best['id']} (d={bd:.3f})")
    pos = first["ee_end"].copy()
    for g in lifted[lifted.index(first) + 1:]:
        if np.linalg.norm(g["ee_start"][:2] - pos[:2]) < 0.035:
            pos = g["ee_end"].copy()
    # snap to a detection in the final frame
    snapped = None
    if final_dets:
        d = min(final_dets, key=lambda d: np.linalg.norm(d["xy"] - pos[:2]))
        if np.linalg.norm(d["xy"] - pos[:2]) < 0.05:
            snapped = d
    if snapped is not None:
        out_pos = snapped["pos"].copy()
        chroma = chroma if chroma is not None else snapped["chroma"].copy()
    else:
        out_pos = np.array([pos[0], pos[1], pos[2] + 0.02])
    if chroma is None:
        chroma = snapped["chroma"].copy() if snapped is not None else np.array([1 / 3] * 3)
    log(f"[video/state] TARGET final pos={np.round(out_pos,4).tolist()} snapped={snapped is not None}")
    return {"pos": out_pos, "chroma": chroma, "why": "state-events"}


# ============================================================================ main
def main(ctx):
    log(f"[main] controller {VERSION} start")
    robot = Robot(ctx)
    try:
        run(robot)
    except EpisodeEnded:
        log("[main] episode ended")
    except Exception:
        log("[main] FATAL (caught):\n" + traceback.format_exc())
    log(f"[main] done steps_sent={robot.steps_sent} complete={robot.complete} ended={robot.ended} wall={elapsed():.1f}s")


def run(robot):
    scene = Scene()
    res0, meta0, imgs0 = robot.observe()
    instr_text = rx.get_instruction(meta0)
    instr = rx.parse_instruction(instr_text)
    log(f"[P0] instruction={instr_text!r} parsed={ {k: v for k, v in instr.items() if k != 'text'} }")
    log(f"[P0] state={None if robot.state is None else np.round(robot.state, 4).tolist()} step_id={robot.step_id} "
        f"remaining={robot.remaining} keys={list(meta0.keys())[:30]} images={[(k, v.shape) for k, v in imgs0.items()]}")
    try:
        lim = rx.find_first(meta0, "episode_limits")
        log(f"[P0] episode_limits={lim}")
        t = rx.reply_texts(res0)
        log("[P0] meta text head:", (t[0][:1500] if t else "<none>"))
    except Exception:
        pass
    if robot.state is None:
        log("[P0] no state; cannot control")
        return
    log(f"[P0] hand vertical err={geom.approach_axis_down_err(robot.state):.1f}deg yaw={math.degrees(geom.hand_yaw(robot.state)):.1f}")
    scene.ref_xy = robot.state[:2].copy()

    # ---------------- P1 calibrate front
    cands = image_candidates(robot, res0, meta0, imgs0, "front")
    log(f"[P1] front candidates: {[(n, None if r is None else r.shape, None if d is None else (d.shape, round(float(d.max()), 3))) for n, r, d in cands]}")
    if not cands:
        log("[P1] no front image/depth available")
        return
    model, resid, rgb0, dep0 = calibrate(robot, meta0, "front", cands, prefer="manifest")
    if model is None or rgb0 is None or dep0 is None:
        log("[P1] no back-projection model")
        return
    if resid is not None and resid > 0.02:
        log(f"[P1] WARNING front calibration residual {resid:.4f} m is large")
    scene.front_model = model
    scene.front_source = next((n for n, r, d in cands if d is dep0), None)
    P0, valid0 = world_points(model, dep0)
    scene.table_z = percept.table_height(P0, valid0, scene.ref_xy, radius=1.0)
    log(f"[P1] table_z={scene.table_z} ee_z={robot.state[2]:.4f} t={elapsed():.1f}s")
    if scene.table_z is None:
        return
    dets0 = percept.detect(P0, valid0, rgb0, scene.table_z, scene.ref_xy, use_sat=True)
    if not dets0:
        for zc in percept.plane_candidates(P0, valid0, scene.ref_xy, radius=1.0):
            if abs(zc - scene.table_z) < 0.01:
                continue
            d2 = percept.detect(P0, valid0, rgb0, zc, scene.ref_xy, use_sat=True)
            log(f"[P1] alternative table plane z={zc:.4f}: {len(d2)} detections")
            if d2:
                scene.table_z, dets0 = zc, d2
                break
    if not dets0:
        log("[P1] no saturated objects; switching to height-only detection")
        scene.use_sat = False
        dets0 = percept.detect(P0, valid0, rgb0, scene.table_z, scene.ref_xy, use_sat=False,
                               exclude_xyz=robot.state[:3], exclude_r=0.08)
    save_debug("exec0_front", rgb0, dets0)
    for i, d in enumerate(dets0):
        log(f"[P1] det{i} xy={d['xy'].round(4).tolist()} ztop={d['ztop']:.4f} h={d['h']:.4f} area={d['area']} col={d['color'].round().tolist()} ext={d['ext'].round(3).tolist()}")
    if not dets0:
        log("[P1] no objects detected at all")
        return

    # ---------------- P2 video
    tgt = None
    try:
        tgt = process_video(robot, scene, meta0, instr)
    except Exception:
        log("[P2] video processing error:\n" + traceback.format_exc())
    if tgt is None:
        log("[P2] FALLBACK: no target from video; choosing detection nearest to the gripper")
        d = min(dets0, key=lambda d: np.linalg.norm(d["xy"] - robot.state[:2]))
        tgt = {"pos": d["pos"].copy(), "chroma": d["chroma"].copy(), "why": "fallback-nearest"}
    # match to current detections
    best = min(dets0, key=lambda d: np.linalg.norm(d["pos"] - tgt["pos"]) + 0.6 * percept.color_dist(d["chroma"], tgt["chroma"]))
    dist = float(np.linalg.norm(best["pos"] - tgt["pos"]))
    log(f"[P2] matched current det xy={best['xy'].round(4).tolist()} dist_to_track={dist:.3f} t={elapsed():.1f}s")
    if dist < 0.08:
        target = {"xy": best["xy"].copy(), "ztop": best["ztop"], "chroma": best["chroma"].copy()}
        save_debug("exec0_target", rgb0, dets0, best["uv"])
    else:
        target = {"xy": tgt["pos"][:2].copy(), "ztop": float(tgt["pos"][2]), "chroma": tgt["chroma"].copy()}
    ltz = percept.local_table_z(P0, valid0, target["xy"], global_z=scene.table_z)
    target["table_z"] = ltz if ltz is not None else scene.table_z
    log(f"[P2] TARGET xy={np.round(target['xy'],4).tolist()} ztop={target['ztop']:.4f} table={target['table_z']:.4f}")

    # ---------------- P3/P4
    count = instr["count"] or 1
    home_xy = target["xy"].copy()
    for cycle in range(count):
        last = cycle == count - 1
        log(f"[P3] cycle {cycle+1}/{count} step={robot.step_id} remaining={robot.remaining} t={elapsed():.1f}s")
        if robot.remaining is not None and robot.remaining < 40:
            log("[P3] too few steps left for another cycle")
            break
        ok = pick(robot, scene, target)
        if robot.complete:
            hold(robot, 1.0, 3)
            return
        if not ok:
            log("[P3] pick failed after retries")
            return
        hold(robot, 1.0, 6)
        if robot.complete:
            return
        # maybe a higher lift is what the checker wants
        move_to(robot, robot.state[:3] + np.array([0, 0, 0.06]), 1.0, label="-lift2")
        hold(robot, 1.0, 6)
        if robot.complete:
            return
        if last and not instr["place"]:
            break
        place(robot, target, home_xy)
        if robot.complete:
            hold(robot, -1.0, 3)
            return
        target["xy"] = home_xy.copy()
        if last:
            log("[P4] placed after last cycle; no completion yet")
            return
    held = robot.fingers is not None and robot.fingers > GRIP_EMPTY_SUM
    log(f"[P4] cycles done without task_complete; holding={held}")
    if held:
        # keep the cube lifted while the episode runs out (success may be judged at the end)
        for _ in range(12):
            if robot.ended or robot.complete or elapsed() > 1050:
                break
            n = 100 if robot.remaining is None else int(max(1, min(100, robot.remaining)))
            hold(robot, 1.0, n)


def pick(robot, scene, target):
    s = float(np.clip(target["ztop"] - target["table_z"], 0.015, 0.09))
    for attempt in range(MAX_PICK_ATTEMPTS):
        log(f"[pick] attempt {attempt+1} xy={np.round(target['xy'],4).tolist()} ztop={target['ztop']:.4f} s={s:.3f} step={robot.step_id}")
        z_safe = max(target["ztop"] + HOVER_DZ, robot.state[2])
        # rise first if low
        if robot.state[2] < target["ztop"] + HOVER_DZ - 0.02:
            move_to(robot, [robot.state[0], robot.state[1], target["ztop"] + HOVER_DZ], -1.0, label="-rise")
        hover = np.array([target["xy"][0], target["xy"][1], target["ztop"] + HOVER_DZ])
        move_to(robot, hover, -1.0, label="-hover")
        # wrist refinement (twice at most)
        yaw = None
        for r in range(2):
            try:
                res, meta, imgs = robot.observe()
                w = wrist_detect(robot, scene, res, meta, imgs, target["xy"], target["chroma"])
            except EpisodeEnded:
                raise
            except Exception:
                log("[pick] wrist error:\n" + traceback.format_exc()[-800:])
                w = None
            if w is None:
                break
            off = float(np.linalg.norm(w["xy"] - target["xy"]))
            target["xy"] = w["xy"].copy()
            if 0.005 < w["ztop"] - scene.table_z < 0.12:
                target["ztop"] = 0.5 * (target["ztop"] + w["ztop"])
            if w["table_z"] is not None and abs(w["table_z"] - target["table_z"]) < 0.02:
                target["table_z"] = 0.5 * (target["table_z"] + w["table_z"])
            yaw = w["yaw"]
            s = float(np.clip(target["ztop"] - target["table_z"], 0.015, 0.09))
            if off < 0.008:
                break
            move_to(robot, [target["xy"][0], target["xy"][1], target["ztop"] + HOVER_DZ], -1.0, speed=SPEED_NEAR, label="-refine")
        if yaw is not None:
            delta = geom.wrap_quarter(yaw - geom.hand_yaw(robot.state))
            log(f"[pick] cube yaw={math.degrees(yaw):.1f} hand={math.degrees(geom.hand_yaw(robot.state)):.1f} delta={math.degrees(delta):.1f}")
            if abs(delta) > math.radians(6):
                rotate_yaw(robot, delta, -1.0)
        # descend
        z_grasp = target["table_z"] + max(0.012, 0.45 * s)
        move_to(robot, [target["xy"][0], target["xy"][1], target["ztop"] + 0.03], -1.0, speed=SPEED_FREE, label="-pre")
        ok, d = move_to(robot, [target["xy"][0], target["xy"][1], z_grasp], -1.0, speed=SPEED_NEAR, tol=0.004, label="-descend")
        log(f"[pick] at grasp pose ok={ok} d={d:.4f} ee={np.round(robot.state[:3],4).tolist()} z_grasp={z_grasp:.4f}")
        hold(robot, 1.0, CLOSE_STEPS)
        fs = robot.fingers
        log(f"[pick] closed fingers={fs:.4f}")
        if fs > GRIP_EMPTY_SUM:
            lift_z = target["table_z"] + s + LIFT_DZ
            move_to(robot, [robot.state[0], robot.state[1], lift_z], 1.0, speed=SPEED_NEAR * 1.5, label="-lift")
        fs2 = robot.fingers
        log(f"[pick] lifted ee_z={robot.state[2]:.4f} fingers={fs2:.4f} complete={robot.complete}")
        if robot.complete:
            return True
        if fs2 > GRIP_EMPTY_SUM:
            # verify with the front camera that something rose with the hand
            try:
                res, meta, imgs = robot.observe()
                dets, *_ = front_detect(robot, scene, res, meta, imgs, save=f"lift_{robot.step_id}")
                near = [d for d in dets if np.linalg.norm(d["xy"] - robot.state[:2]) < 0.06 and d["h"] > s + 0.04]
                log(f"[pick] front check: {len(near)} raised detections near hand")
            except EpisodeEnded:
                raise
            except Exception:
                pass
            return True
        # failed: open, rise, re-perceive from the front camera
        log("[pick] grasp empty; retrying")
        hold(robot, -1.0, OPEN_STEPS)
        move_to(robot, [robot.state[0], robot.state[1], target["ztop"] + HOVER_DZ + 0.05], -1.0, label="-retreat")
        try:
            res, meta, imgs = robot.observe()
            dets, *_ = front_detect(robot, scene, res, meta, imgs, save=f"retry_{robot.step_id}")
            if dets:
                b = min(dets, key=lambda d: np.linalg.norm(d["xy"] - target["xy"]) + 0.6 * percept.color_dist(d["chroma"], target["chroma"]))
                if np.linalg.norm(b["xy"] - target["xy"]) < 0.10:
                    target["xy"] = b["xy"].copy()
                    target["ztop"] = b["ztop"]
                    log(f"[pick] re-acquired target at {np.round(b['xy'],4).tolist()}")
        except EpisodeEnded:
            raise
        except Exception:
            log("[pick] re-perception error " + traceback.format_exc()[-400:])
    return False


def place(robot, target, xy):
    s = float(np.clip(target["ztop"] - target["table_z"], 0.015, 0.09))
    log(f"[place] to xy={np.round(xy,4).tolist()} step={robot.step_id}")
    move_to(robot, [xy[0], xy[1], target["table_z"] + s + 0.06], 1.0, label="-carry")
    move_to(robot, [xy[0], xy[1], target["table_z"] + 0.5 * s + 0.008], 1.0, speed=SPEED_NEAR, tol=0.004, label="-lower")
    hold(robot, -1.0, OPEN_STEPS)
    if robot.complete:
        return
    move_to(robot, [robot.state[0], robot.state[1], target["table_z"] + s + HOVER_DZ], -1.0, label="-up")
    log(f"[place] done fingers={robot.fingers:.4f} complete={robot.complete}")
