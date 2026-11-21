"""VideoPlaceButton controller (RoboMME-Plus, demo-only, developed offline only).

1. Read the conditioning video (frame_sequence manifest) and decide the goal target:
   the target on which the instructed-colour cube was first released after the button
   press, followed through all later target swaps (videoan.analyse).
2. Localise the instructed cube and the goal target with depth back-projection.
3. Pick (hover, wrist refinement, yaw alignment, descend, close, lift, verify) and
   place (transit, descend, open, lift, verify); retry while steps and time remain.
Never raises: every phase is wrapped and logged to /workspace/output/vpb_log.txt.
"""
import glob
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import perception as P   # noqa: E402
import videoan           # noqa: E402
import mcpio as M        # noqa: E402
import geom as G         # noqa: E402

WS = Path("/workspace")
OUT = WS / "output"
WALL_LIMIT_S = 1050.0           # controller hard limit is 1200 s; keep margin
DEFAULT_STEP_LIMIT = 400        # used only when the packet does not report a limit
HOVER_DZ = 0.13                 # TCP above table while hovering over cube / target
TRANSIT_DZ = 0.17               # TCP above table while carrying
CUBE_H_DEFAULT = 0.04
PLACE_CLEAR = 0.012             # cube bottom above target top at release
V_FAST, V_SLOW = 0.4, 0.18      # action units (x 0.05 m per step)


class Log:
    def __init__(self):
        self.t0 = time.time()
        self.f = None
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / "vpb_log.txt", "a")
        except Exception:
            self.f = None

    def __call__(self, *a):
        s = "[%7.2f] " % (time.time() - self.t0) + " ".join(str(x) for x in a)
        try:
            print(s, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(s + "\n")
                self.f.flush()
            except Exception:
                pass

    def save_img(self, name, arr):
        try:
            Image.fromarray(np.asarray(arr, np.uint8)).save(OUT / name)
        except Exception:
            pass


class EpisodeOver(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx, self.log = ctx, log
        self.state = None
        self.step_id = None
        self.meta = {}
        self.imgs = {}
        self.ended = False
        self.steps_sent = 0
        self.remaining = None
        self.calib = {}
        self.calib_ok = {}
        self.rgb_map = None      # (front index, wrist index) among the reply's RGB image blocks
        self.raw_rgbs = []

    # ---------- MCP ----------
    def _absorb(self, res):
        meta = M.meta_of(res)
        self.meta = meta
        s = M.state_of(meta)
        if s is not None:
            self.state = s
        sid = M.step_id_of(meta)
        if sid is not None:
            self.step_id = sid
        r = M.remaining_steps_of(meta)
        if r is None:
            try:
                lim = M.find(meta, "episode_limits", lambda v: isinstance(v, dict)) or {}
                mx = next((v for k, v in lim.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and ("max" in k or "limit" in k) and "step" in k), None)
                if mx is not None and self.step_id is not None:
                    r = int(mx - self.step_id)
            except Exception:
                r = None
        if r is not None:
            self.remaining = r
        if M.ended_of(meta):
            self.ended = True
        ims = M.split_images(res, meta)
        if "front" not in ims:
            ims = self._images_from_manifest(meta) or ims
        try:
            self.raw_rgbs = [M.to_rgb_array(im) for im, _, _ in M.images_of(res) if not M.is_depth_image(im)]
        except Exception:
            self.raw_rgbs = []
        if self.rgb_map is not None and len(self.raw_rgbs) > max(self.rgb_map):
            ims = dict(ims)
            ims["front"] = self.raw_rgbs[self.rgb_map[0]]
            ims["wrist"] = self.raw_rgbs[self.rgb_map[1]]
        if ims:
            self.imgs = ims
        self._calib(meta)
        return meta

    def _images_from_manifest(self, meta):
        try:
            mp = M.manifest_path_of(meta)
            if not mp:
                return {}
            p = WS / mp if not os.path.isabs(mp) else Path(mp)
            man = json.loads(p.read_text())
            fr = man.get("frames") or []
            if not fr:
                return {}
            f = max(fr, key=lambda f: f.get("frame_index", 0))
            out = {}
            for cam in ("front", "wrist"):
                q = frame_file(f, cam, "rgb", p)
                if q:
                    out[cam] = load_rgb(q)
                q = frame_file(f, cam, "depth", p)
                if q:
                    with Image.open(q) as im:
                        out[cam + "_depth"] = np.asarray(im, np.float64) * 0.001
            return {k: v for k, v in out.items() if v is not None}
        except Exception:
            return {}

    def _calib(self, meta):
        try:
            Ks = M.find(meta, "camera_intrinsics")
            Ps = M.find(meta, "camera_poses")
            if not isinstance(Ks, dict) or not isinstance(Ps, dict):
                return
            for cam in ("front", "wrist"):
                kk = next((v for k, v in Ks.items() if cam in k), None)
                pp = next((v for k, v in Ps.items() if cam in k), None)
                K = G.parse_K(kk) if kk is not None else None
                pose = G.parse_pose(pp) if pp is not None else None
                if K is not None and pose is not None:
                    self.calib[cam] = G.Calib(K, pose[0], pose[1], pose[2])
        except Exception:
            pass

    def observe(self):
        res = self.ctx.call("robomme_observe")
        return res, self._absorb(res)

    def time_left(self):
        return WALL_LIMIT_S - (time.time() - self.log.t0)

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        return DEFAULT_STEP_LIMIT - self.steps_sent

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        i = 0
        while i < len(rows):
            if self.ended:
                raise EpisodeOver()
            if self.time_left() < 15:
                raise EpisodeOver("wall time")
            chunk = rows[i:i + 100]
            sl = self.steps_left()
            if sl is not None and sl <= 0:
                raise EpisodeOver("steps")
            if sl is not None and len(chunk) > sl:
                chunk = chunk[:max(1, sl)]
            try:
                res = self.ctx.call("robomme_step", actions=chunk)
            except Exception as e:  # do not retry an uncertain motion: resync
                self.log("robomme_step error:", repr(e)[:300])
                self.steps_sent += len(chunk)
                try:
                    st = self.ctx.call("robomme_status")
                    self._absorb(st)
                except Exception:
                    pass
                try:
                    self.observe()
                except Exception as e2:
                    self.log("observe after error failed:", repr(e2)[:200])
                    raise EpisodeOver("tool failure")
                if self.ended:
                    raise EpisodeOver()
                i += len(chunk)
                continue
            self.steps_sent += len(chunk)
            self._absorb(res)
            if self.state is None:
                self.observe()
            i += len(chunk)
        if self.ended:
            raise EpisodeOver()

    # ---------- motion ----------
    def pos(self):
        return np.asarray(self.state[:3], float)

    def fingers(self):
        return float(self.state[6] + self.state[7])

    def move_to(self, goal, grip, vmax=V_FAST, tol=0.004, iters=4, settle=2, yaw_goal=None):
        goal = np.asarray(goal, float)
        for _ in range(iters):
            e = goal - self.pos()
            d = float(np.linalg.norm(e))
            if d < tol:
                return True
            n = int(math.ceil(d / (vmax * 0.05)))
            n = max(1, min(n, 80))
            row = list(e / n / 0.05) + [0.0, 0.0, 0.0, grip]
            self.step([row] * n + [[0, 0, 0, 0, 0, 0, grip]] * settle)
        return float(np.linalg.norm(goal - self.pos())) < max(tol * 3, 0.012)

    def hold(self, n, grip):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def rotate_yaw(self, delta, grip, hold_pos):
        """Rotate about world z by delta (|delta| <= 45 deg), verified mod 90 deg by the measured yaw."""
        target = G.gripper_yaw(self.state[3:6]) + delta
        sign = 1.0
        for it in range(3):
            cur = G.gripper_yaw(self.state[3:6])
            err = G.wrap90(target - cur)
            if abs(err) < math.radians(3):
                return True
            n = max(1, int(math.ceil(abs(err) / 0.12)))
            rows = [[0, 0, 0, 0, 0, sign * err / n / 0.5, grip] for _ in range(n)]
            self.step(rows + [[0, 0, 0, 0, 0, 0, grip]] * 2)
            moved = G.wrap90(G.gripper_yaw(self.state[3:6]) - cur)
            self.log("rotate it=%d err=%.1fdeg moved=%.1fdeg" % (it, math.degrees(err), math.degrees(moved)))
            if abs(err) > math.radians(6) and moved * err < 0:
                sign = -sign
            e = np.asarray(hold_pos) - self.pos()
            if np.linalg.norm(e) > 0.006:
                self.move_to(hold_pos, grip, vmax=V_SLOW, iters=1)
        return False

    # ---------- localisation ----------
    def points(self, cam, pix, radius=1):
        """World points for pixel list [(u, v)] on the current observation."""
        pix = [(float(u), float(v)) for u, v in pix]
        out = []
        ok = True
        for i in range(0, len(pix), 32):
            sel = [{"camera": cam, "u": u, "v": v} for u, v in pix[i:i + 32]]
            try:
                res = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(self.step_id or 0),
                                    selections=sel, depth_window_radius=radius)
                pts = M.ptp_points(res, len(sel))
            except Exception as e:
                self.log("ptp error", repr(e)[:200])
                pts = None
            if pts is None:
                ok = False
                break
            out.extend(p for p, _ in pts)
        if ok:
            return [p for p in out if p is not None]
        own = self.own_points(cam, pix)
        if own:
            self.log("using own back-projection for", cam, len(own))
        return own

    def own_points(self, cam, pix):
        c = self.calib.get(cam)
        d = self.imgs.get(cam + "_depth")
        if c is None or d is None:
            return []
        out = []
        for u, v in pix:
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iv < d.shape[0] and 0 <= iu < d.shape[1] and d[iv, iu] > 0:
                out.append(c.backproject(np.array([u]), np.array([v]), np.array([d[iv, iu]]))[0])
        return out


# ---------------------------------------------------------------- helpers

def sample_pixels(mask, n=32, erode=True):
    from scipy import ndimage
    m = mask
    if erode:
        e = ndimage.binary_erosion(mask)
        if e.sum() >= min(n, 8):
            m = e
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    idx = np.unique(np.round(np.linspace(0, len(xs) - 1, min(n, len(xs)))).astype(int))
    return [(xs[i] + 0.5 * 0, ys[i]) for i in idx]


def top_face(pts, band=0.007):
    a = np.asarray(pts, float)
    if len(a) == 0:
        return None, None, a
    zt = float(np.percentile(a[:, 2], 85))
    top = a[a[:, 2] > zt - band]
    if len(top) == 0:
        top = a
    return np.median(top[:, :2], axis=0), float(np.median(top[:, 2])), top


def instruction_color(text):
    t = (text or "").lower()
    m = re.search(r"\b(red|green|blue)\s+cube", t)
    if m:
        return m.group(1)
    for c in ("red", "blue", "green"):
        if c in t:
            return c
    return None


def load_manifest(meta, log):
    paths = []
    mp = M.manifest_path_of(meta)
    if mp:
        paths.append(WS / mp if not os.path.isabs(mp) else Path(mp))
    cands = sorted(glob.glob(str(WS / "runtime/autonomous_controller/runs/*/frames/**/manifest.json"),
                         recursive=True), key=lambda p: -os.path.getmtime(p))
    paths.extend(Path(p) for p in cands[:20])
    for p in paths:
        try:
            man = json.loads(p.read_text())
            frames = man.get("frames") or []
            if any(f.get("phase") == "video_demo" for f in frames):
                log("manifest", p, "frames", len(frames))
                return man, p
        except Exception as e:
            log("manifest read failed", p, repr(e)[:200])
    return None, None


def frame_file(f, cam, kind, mpath):
    for e in f.get("files", []):
        if e.get("camera") == cam and e.get("kind") == kind:
            p = e.get("path")
            if not p:
                continue
            for base in (WS, mpath.parent if mpath else WS):
                q = base / p
                if q.exists():
                    return q
    return None


def load_rgb(p):
    try:
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), np.uint8)
    except Exception:
        return None


def load_video(man, mpath, log):
    frames = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: (f.get("frame_index", 0)))
    out = []
    for f in frames:
        fp = frame_file(f, "front", "rgb", mpath)
        wp = frame_file(f, "wrist", "rgb", mpath)
        fr = load_rgb(fp) if fp else None
        if fr is None:
            continue
        wr = load_rgb(wp) if wp else None
        out.append((float(f.get("timestamp_s", len(out) * 0.05)), fr, wr))
    ex = [f for f in man.get("frames", []) if f.get("phase") == "execution"]
    exec_imgs = {}
    if ex:
        f = sorted(ex, key=lambda f: f.get("frame_index", 0))[0]
        for cam in ("front", "wrist"):
            p = frame_file(f, cam, "rgb", mpath)
            if p:
                exec_imgs[cam] = load_rgb(p)
    log("video frames", len(out), "exec imgs", list(exec_imgs))
    return out, exec_imgs


# ---------------------------------------------------------------- task logic

class Task:
    def __init__(self, ctx):
        self.log = Log()
        self.R = Robot(ctx, self.log)
        self.color = None
        self.goal_px = None
        self.goal_mask = None
        self.goal_xyz = None
        self.table_z = None
        self.cube_h = CUBE_H_DEFAULT
        self.home = None
        self.grasp_off = None

    # ----- decision from video
    def decide(self, meta):
        log = self.log
        self.color = instruction_color(M.instruction_of(meta)) or "red"
        log("instruction:", M.instruction_of(meta), "-> color", self.color)
        man, mpath = load_manifest(meta, log)
        if man is None:
            log("NO MANIFEST: fallback goal")
            return None
        video, exec_imgs = load_video(man, mpath, log)
        # identify front/wrist among the reply's RGB images using the manifest's execution frame
        try:
            ef, ew = exec_imgs.get("front"), exec_imgs.get("wrist")
            raws = [r for r in self.R.raw_rgbs if r.shape[:2] == (ef.shape[:2] if ef is not None else r.shape[:2])]
            if ef is not None and ew is not None and len(raws) == len(self.R.raw_rgbs) and len(raws) >= 2:
                df = [float(np.abs(r.astype(float) - ef).mean()) for r in raws]
                dw = [float(np.abs(r.astype(float) - ew).mean()) for r in raws]
                fi = int(np.argmin(df))
                wi = int(np.argmin([d if i != fi else 1e9 for i, d in enumerate(dw)]))
                log("cam check: dist to exec front %s, to exec wrist %s" % (np.round(df, 1), np.round(dw, 1)))
                if df[fi] < 20 and dw[wi] < 20 and (fi, wi) != (0, 1):
                    log("camera mapping corrected to", (fi, wi))
                    self.R.rgb_map = (fi, wi)
                    self.R.observe()
        except Exception as e:
            log("cam check failed", repr(e))
        if not video:
            return None
        res = videoan.analyse(video, self.color)
        for line in res["log"]:
            log("  va:", line)
        log("analysis confidence", res.get("confidence"))
        self.slots = res.get("slots")
        if res.get("slot_xy") is not None:
            k = res["slot"]
            return self.slots[k]
        return None

    # ----- localisation
    def front(self):
        self.R.observe()
        return self.R.imgs.get("front")

    def locate_table_near(self, front, x, y, r0=10, r1=24):
        tw, wood = P.tw_wood(front)
        H, W = wood.shape
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.hypot(xx - x, yy - y)
        m = wood & (rr > r0) & (rr < r1) & ~P.color_masks(front)["red"] & ~P.color_masks(front)["blue"] \
            & ~P.color_masks(front)["green"]
        pix = sample_pixels(m, 16, erode=False)
        pts = self.R.points("front", pix, radius=0)
        if len(pts) < 3:
            return None
        return float(np.median(np.asarray(pts)[:, 2]))

    def locate_cube_front(self):
        front = self.front()
        if front is None:
            return None
        b = P.cube_blob(front, self.color, min_pixels=6)
        if b is None:
            self.log("cube not visible in front view")
            return None
        x, y, n, mask = b
        pts = self.R.points("front", sample_pixels(mask, 32), radius=0)
        if len(pts) < 3:
            self.log("cube points failed", len(pts))
            return None
        xy, zt, top = top_face(pts)
        tz = self.locate_table_near(front, x, y)
        if tz is not None:
            self.table_z = tz if self.table_z is None else self.table_z
        self.log("cube px (%.0f,%.0f) n=%d -> xy %s top z %.3f table %s" % (x, y, n, np.round(xy, 3), zt, tz))
        return np.array([xy[0], xy[1], zt]), (x, y)

    def locate_goal(self, slot):
        front = self.front()
        tw, wood = P.tw_wood(front)
        foot = slot["foot"] if slot is not None else None
        pts = []
        if foot is not None:
            cm = P.color_masks(front)
            clean = foot & ~(cm["red"] | cm["green"] | cm["blue"])
            m = clean & tw
            if m.sum() < 10:
                m = clean
            pts = self.R.points("front", sample_pixels(m, 32, erode=False), radius=0)
        if len(pts) < 3:
            return None
        a = np.asarray(pts)
        z = float(np.median(a[:, 2]))
        a = a[np.abs(a[:, 2] - z) < 0.02]
        xy = a[:, :2].mean(axis=0)
        self.log("goal slot px %s -> xyz (%.3f, %.3f, %.3f) n=%d" % (np.round(slot["xy"]), xy[0], xy[1], z, len(a)))
        return np.array([xy[0], xy[1], z])

    def wrist_refine(self, guess_xy):
        """Cube xy/top z/yaw from the wrist camera at hover; None if inconsistent."""
        self.R.observe()
        w = self.R.imgs.get("wrist")
        if w is None:
            return None
        self.log.save_img("wrist_hover.png", w)
        m = P.color_masks(w)[self.color]
        comps = P.components(m, 30)
        if not comps:
            self.log("wrist: cube not seen")
            return None
        best = None
        for x, y, c, reg in sorted(comps, key=lambda t: -t[2])[:3]:
            pix = sample_pixels(reg, 64)
            pts = self.R.points("wrist", pix, radius=0)
            if len(pts) < 8:
                continue
            xy, zt, top = top_face(pts, band=0.006)
            dist = float(np.hypot(*(xy - guess_xy)))
            if best is None or dist < best[0]:
                best = (dist, xy, zt, top)
        if best is None:
            return None
        dist, xy, zt, top = best
        yaw, contrast = G.square_yaw(top[:, :2]) if len(top) >= 12 else (None, 0.0)
        self.log("wrist refine xy %s (d=%.3f from front) top z %.3f yaw %s contrast %.2f n_top %d" % (
            np.round(xy, 3), dist, zt, None if yaw is None else round(math.degrees(yaw), 1), contrast, len(top)))
        if dist > 0.035:
            return None
        return xy, zt, (yaw if contrast > 0.06 else None)

    # ----- pick and place
    def pick(self):
        R, log = self.R, self.log
        loc = self.locate_cube_front()
        if loc is None:
            return False
        cube, _ = loc
        tz = self.table_z if self.table_z is not None else cube[2] - CUBE_H_DEFAULT
        h = cube[2] - tz
        if not (0.015 < h < 0.08):
            log("cube height %.3f implausible; using default" % h)
            h = CUBE_H_DEFAULT
            tz = cube[2] - h
        self.cube_h = h
        hover_z = tz + HOVER_DZ
        p = R.pos()
        if p[2] < hover_z - 0.02:
            R.move_to([p[0], p[1], hover_z], -1, vmax=V_FAST, iters=2)
        R.move_to([cube[0], cube[1], hover_z], -1, vmax=V_FAST, iters=3)
        xy = cube[:2]
        ref = self.wrist_refine(xy)
        yaw = None
        if ref is not None:
            xy, zt, yaw = ref
            if abs((zt - tz) - h) < 0.015:
                pass
        if yaw is not None and G.pointing_down(R.state[3:6]):
            gy = G.gripper_yaw(R.state[3:6])
            delta = G.wrap90(yaw - gy)
            log("yaw align: cube %.1f gripper %.1f delta %.1f" % (math.degrees(yaw), math.degrees(gy),
                                                                  math.degrees(delta)))
            if abs(delta) > math.radians(5):
                R.rotate_yaw(delta, -1, [xy[0], xy[1], hover_z])
        elif yaw is None:
            log("yaw align skipped")
        # descend
        pre_z = tz + h + 0.035
        R.move_to([xy[0], xy[1], pre_z], -1, vmax=V_FAST, iters=3, tol=0.003)
        zg = tz + max(0.5 * h, 0.016)
        R.move_to([xy[0], xy[1], zg], -1, vmax=V_SLOW, iters=3, tol=0.003)
        log("at grasp pose", np.round(R.pos(), 3), "target z %.3f" % zg)
        R.hold(8, 1.0)
        prev = R.fingers()
        for _ in range(4):           # wait until the fingers stop moving
            R.hold(3, 1.0)
            f_now = R.fingers()
            if abs(f_now - prev) < 0.0015:
                break
            prev = f_now
        f_closed = R.fingers()
        self.grasp_off = R.pos()[2] - tz
        R.move_to([R.pos()[0], R.pos()[1], tz + TRANSIT_DZ], 1.0, vmax=V_SLOW + 0.1, iters=2, settle=1)
        R.observe()
        f = R.fingers()
        w = R.imgs.get("wrist")
        held = P.held_color(w) if w is not None else None
        log("after lift fingers %.4f (at close %.4f) wrist held=%s ee %s" % (f, f_closed, held, np.round(R.pos(), 3)))
        if w is not None:
            self.log.save_img("wrist_lift.png", w)
        ok = f > 0.008 and (held == self.color or held is None) and f < 0.077
        if held is not None and held != self.color:
            ok = False
        if not ok:
            log("grasp failed; opening")
            R.hold(8, -1.0)
        return ok

    def place(self):
        R, log = self.R, self.log
        g = self.goal_xyz
        tz = self.table_z if self.table_z is not None else g[2]
        R.move_to([g[0], g[1], tz + TRANSIT_DZ], 1.0, vmax=V_FAST, iters=3, settle=2)
        off = self.grasp_off if self.grasp_off is not None else 0.5 * self.cube_h
        zp = g[2] + off + PLACE_CLEAR
        R.move_to([g[0], g[1], zp], 1.0, vmax=V_SLOW, iters=3, tol=0.003)
        log("release at", np.round(R.pos(), 3), "goal", np.round(g, 3))
        R.hold(10, -1.0)
        p = R.pos()
        R.move_to([p[0], p[1], tz + HOVER_DZ], -1.0, vmax=V_SLOW + 0.1, iters=2)

    def retreat(self):
        if self.home is not None:
            try:
                self.R.move_to(self.home, -1.0, vmax=V_FAST, iters=2)
            except EpisodeOver:
                raise
            except Exception as e:
                self.log("retreat failed", repr(e))

    def verify(self):
        loc = self.locate_cube_front()
        if loc is None:
            return None
        cube, _ = loc
        d = float(np.hypot(*(cube[:2] - self.goal_xyz[:2])))
        self.log("verify: cube-goal xy distance %.3f, cube top z %.3f" % (d, cube[2]))
        return d

    # ----- main
    def run(self):
        R, log = self.R, self.log
        res, meta = R.observe()
        log("initial keys", list(meta.keys())[:30], "step", R.step_id, "state", R.state)
        try:
            log("limits", json.dumps(M.find(meta, "episode_limits"))[:300])
        except Exception:
            pass
        if R.state is None:
            log("no state in observation; abort")
            return
        self.home = R.pos().copy()
        for k, v in R.imgs.items():
            if k in ("front", "wrist"):
                log.save_img("start_%s.png" % k, v)
        slot = None
        try:
            slot = self.decide(meta)
        except Exception:
            log("decide failed", traceback.format_exc()[-1500:])
        if slot is None:
            slot = self.fallback_slot()
        if slot is None:
            log("no goal found; abort")
            return
        try:
            self.goal_xyz = self.locate_goal(slot)
        except Exception:
            log("goal localisation failed", traceback.format_exc()[-800:])
        if self.goal_xyz is None:
            log("goal localisation failed; abort")
            return
        if self.table_z is None:
            self.table_z = float(self.goal_xyz[2])
        attempts = 0
        while attempts < 4 and R.time_left() > 120 and R.steps_left() > 60:
            attempts += 1
            log("=== attempt", attempts, "steps left", R.steps_left(), "time left %.0f" % R.time_left())
            try:
                if not self.pick():
                    continue
                self.place()
                self.retreat()
                d = self.verify()
                if d is not None and d < 0.02:
                    log("placed on goal; done")
                    return
                if d is None:
                    log("cube not found after place; stopping")
                    return
            except EpisodeOver:
                raise
            except Exception:
                log("attempt error", traceback.format_exc()[-1500:])
                try:
                    R.observe()
                    if R.fingers() < 0.07:
                        R.hold(8, -1.0)
                except EpisodeOver:
                    raise
                except Exception:
                    pass

    def fallback_slot(self):
        try:
            front = self.R.imgs.get("front")
            comps = P.components(P.target_mask(front), P.SLOT_MIN_PIXELS)
            if not comps:
                return None
            x, y, c, reg = max(comps, key=lambda t: t[2])
            self.log("fallback slot at (%.0f,%.0f)" % (x, y))
            return {"xy": np.array([x, y]), "foot": reg, "n": c}
        except Exception:
            return None


def main(ctx):
    t = None
    try:
        t = Task(ctx)
        t.run()
    except EpisodeOver as e:
        if t:
            t.log("episode over:", e)
    except Exception:
        if t:
            t.log("FATAL", traceback.format_exc()[-2000:])
    finally:
        if t:
            t.log("controller end; steps sent", t.R.steps_sent, "ended", t.R.ended)
