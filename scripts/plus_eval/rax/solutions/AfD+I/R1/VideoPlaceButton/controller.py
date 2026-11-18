"""VideoPlaceButton controller (RAX, RoboMME-Plus).

Pipeline / transition table
  S0 observe+status         -> S1 (always; failures logged)
  S1 analyse video          -> S2 with destination pixel; on failure -> fallback destination
  S2 localise cube/table/dest (front camera, pixel_to_position; own back-projection fallback)
  S3 pick attempt (<=3):  hover above cube -> wrist refine xy (+yaw align) -> descend -> close
                          -> verify fingers (not fully closed) -> lift
        fail -> open, re-localise from front, next attempt
  S4 carry to destination hover -> descend -> open -> lift
  S5 verify in front view (cube near destination pixel) -> done / back to S3 if steps remain
Stops whenever the episode reports ended, steps or wall time run low.
"""
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perception as P  # noqa: E402
import rio  # noqa: E402
from rio import log  # noqa: E402

VERSION = "vpb-v1"

# ---------------------------------------------------------------- tunables (untested live; see notes)
POS_SCALE = 0.05          # m per unit action (documented)
ROT_SCALE = 0.5           # rad per unit action (documented)
VMAX_FAST = 0.5           # action units (2.5 cm/step) for free-space moves
VMAX_SLOW = 0.2           # 1 cm/step near objects
HOVER_ABOVE_TOP = 0.12    # m above cube top for wrist inspection
CARRY_ABOVE_TABLE = 0.16  # m
PLACE_CLEARANCE = 0.012   # m cube bottom above table at release
GRASP_MIN_ABOVE_TABLE = 0.012
DEFAULT_CUBE_H = 0.04
FINGER_EMPTY_SUM = 0.006  # m, fingers closed on nothing
WALL_BUDGET_S = 1050.0
MAX_ATTEMPTS = 3
STEP_RESERVE = 5


class EpisodeOver(Exception):
    pass


def rotvec_to_mat(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.t0 = time.time()
        self.state = None
        self.step_id = None
        self.steps = 0
        self.max_steps = None
        self.ended = False
        self.complete = False
        self.front = None
        self.wrist = None
        self.front_ref = None
        self.meta = {}
        self.calls = 0

    # ------------------------------------------------------------ io
    def _check_end(self, meta):
        lim = rio.find_key(meta, "episode_limits")
        if isinstance(lim, dict):
            for k in ("remaining_steps", "steps_remaining", "remaining"):
                v = lim.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.max_steps = self.steps + int(v)
                    break
            else:
                for k in ("max_steps", "step_limit", "max_episode_steps", "limit"):
                    v = lim.get(k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                        self.max_steps = int(v)
                        break
        if rio.flag(meta, "episode_ended") or rio.flag(meta, "task_complete") or rio.flag(meta, "terminated"):
            self.ended = True
            self.complete = rio.flag(meta, "task_complete") or rio.flag(meta, "success")
            log("EPISODE ENDED flag seen; task_complete=", self.complete)

    def wall(self):
        return time.time() - self.t0

    def remaining_steps(self):
        if self.max_steps is None:
            return 10_000
        return self.max_steps - self.steps

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            meta = rio.meta_of(r)
            log("status:", str(meta)[:1500])
            for k in ("max_steps", "step_limit", "max_episode_steps", "horizon"):
                v = rio.find_key(meta, k)
                if isinstance(v, (int, float)) and v > 0:
                    self.max_steps = int(v)
                    break
            used = rio.find_key(meta, "steps_used") or rio.find_key(meta, "elapsed_steps")
            if isinstance(used, (int, float)):
                self.steps = int(used)
            self._check_end(meta)
            return meta
        except Exception as e:
            log("status failed", repr(e))
            return {}

    def _assign_images(self, result, meta):
        imgs = rio.decode_images(result)
        rgb = []
        for label, im in imgs:
            if im.mode in ("RGB", "RGBA", "P"):
                rgb.append((label.lower(), np.array(im.convert("RGB"))))
        front = wrist = None
        for label, a in rgb:
            if "wrist" in label and wrist is None:
                wrist = a
            elif "front" in label and front is None:
                front = a
        if (front is None or wrist is None) and len(rgb) >= 2:
            a0, a1 = rgb[0][1], rgb[1][1]
            if self.front_ref is not None and a0.shape == self.front_ref.shape and a1.shape == self.front_ref.shape:
                d0 = np.abs(a0.astype(int) - self.front_ref).mean()
                d1 = np.abs(a1.astype(int) - self.front_ref).mean()
                front, wrist = (a0, a1) if d0 <= d1 else (a1, a0)
            else:
                front, wrist = a0, a1
        elif front is None and len(rgb) == 1:
            front = rgb[0][1]
        # manifest files (explicit labels) only when inline images are missing
        man, full = (None, None)
        if front is None or wrist is None:
            man, full = rio.manifest_frames(meta)
        if man and man.get("frames"):
            fr = sorted(man["frames"], key=lambda f: f.get("frame_index", 0))[-1]
            try:
                p = rio.frame_file(fr, "front", "rgb", full)
                if front is None and p and os.path.exists(p):
                    front = rio.load_rgb(p)
                p = rio.frame_file(fr, "wrist", "rgb", full)
                if wrist is None and p and os.path.exists(p):
                    wrist = rio.load_rgb(p)
            except Exception as e:
                log("manifest image load failed", repr(e))
        if front is not None:
            self.front = front
        if wrist is not None:
            self.wrist = wrist

    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.calls += 1
        meta = rio.meta_of(r)
        self.meta = meta
        s = rio.get_state(meta)
        if s is not None:
            self.state = s
        sid = rio.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
            self.steps = max(self.steps, sid)
        self._assign_images(r, meta)
        self._check_end(meta)
        return r, meta

    def step(self, rows):
        if self.ended:
            raise EpisodeOver()
        if self.wall() > WALL_BUDGET_S:
            raise EpisodeOver("wall budget")
        rem = self.remaining_steps() - STEP_RESERVE
        if rem <= 0:
            raise EpisodeOver("steps")
        rows = rows[:max(1, min(100, rem))]
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        r = self.ctx.call("robomme_step", actions=rows)
        self.calls += 1
        meta = rio.meta_of(r)
        self.steps += len(rows)
        s = rio.get_state(meta)
        sid = rio.get_step_id(meta)
        if sid is not None:
            self.step_id = sid
            self.steps = max(self.steps, sid)
        self._check_end(meta)
        if s is not None:
            self.state = s
        else:
            if not self.ended:
                self.observe()
        if self.ended:
            raise EpisodeOver()
        return meta

    # ------------------------------------------------------------ geometry
    def pix2world(self, camera, uvs, radius=1):
        """World points for pixel list [(u,v)] of the current observation. None where invalid."""
        if not uvs:
            return []
        out = [None] * len(uvs)
        for start in range(0, len(uvs), 32):
            chunk = uvs[start:start + 32]
            sel = [{"camera": camera, "u": float(u), "v": float(v)} for u, v in chunk]
            try:
                r = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(self.step_id or 0),
                                  selections=sel, depth_window_radius=int(radius))
                meta = rio.meta_of(r)
                pts = rio.parse_points(meta, len(chunk))
                for i, (xyz, info) in enumerate(pts):
                    if xyz is not None:
                        spread = info.get("depth_spread_m") if isinstance(info, dict) else None
                        if isinstance(spread, (int, float)) and spread > 0.03:
                            continue
                        out[start + i] = xyz
                if self.calls < 6 or all(p is None for p in out[start:start + len(chunk)]):
                    log("pix2world reply sample:", str(meta)[:600])
            except Exception as e:
                log("pix2world failed", repr(e))
            self.calls += 1
        if all(p is None for p in out):
            fb = self.backproject(camera, uvs)
            if fb is not None:
                log("using own back-projection fallback")
                return fb
        return out

    def backproject(self, camera, uvs):
        """Fallback: documented conversion from the observation's depth PNG and calibration."""
        try:
            meta = self.meta
            intr = rio.find_key(meta, "camera_intrinsics")
            poses = rio.find_key(meta, "camera_poses")
            if isinstance(intr, dict) and camera in intr:
                intr = intr[camera]
            if isinstance(poses, dict) and camera in poses:
                poses = poses[camera]
            K = None
            if isinstance(intr, dict):
                if all(k in intr for k in ("fx", "fy", "cx", "cy")):
                    K = (intr["fx"], intr["fy"], intr["cx"], intr["cy"])
                else:
                    for k in ("K", "matrix", "intrinsic", "intrinsics", "camera_matrix"):
                        if k in intr:
                            M = np.array(intr[k], float).reshape(3, 3)
                            K = (M[0, 0], M[1, 1], M[0, 2], M[1, 2])
                            break
            elif intr is not None:
                M = np.array(intr, float).reshape(3, 3)
                K = (M[0, 0], M[1, 1], M[0, 2], M[1, 2])
            if K is None or not isinstance(poses, dict):
                return None
            pos = rio.as_vec(poses.get("position_m", poses.get("position")) or poses.get("p"), 3)
            q = rio.as_vec(poses.get("quaternion_wxyz") or poses.get("q"), 4)
            if pos is None or q is None:
                return None
            R = quat_wxyz_to_mat(q)
            man, full = rio.manifest_frames(meta)
            if not man:
                return None
            fr = sorted(man["frames"], key=lambda f: f.get("frame_index", 0))[-1]
            p = rio.frame_file(fr, camera, "depth", full)
            from PIL import Image
            with Image.open(p) as im:
                depth = np.array(im).astype(np.float64) * 0.001
            out = []
            for u, v in uvs:
                ui, vi = int(round(u)), int(round(v))
                win = depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
                win = win[win > 0]
                if win.size == 0:
                    out.append(None)
                    continue
                d = float(np.median(win))
                xc = (u - K[2]) / K[0] * d
                yc = -(v - K[3]) / K[1] * d
                pc = np.array([xc, yc, -d])
                out.append(R @ pc + pos)
            return out
        except Exception as e:
            log("backproject failed", repr(e))
            return None

    # ------------------------------------------------------------ motion
    def pos(self):
        return np.array(self.state[:3], float)

    def fingers(self):
        return float(self.state[6] + self.state[7])

    def move_to(self, goal, grip, tol=0.004, vmax=VMAX_FAST, max_calls=40, label=""):
        goal = np.asarray(goal, float)
        best = 1e9
        stall = 0
        for it in range(max_calls):
            err = goal - self.pos()
            d = float(np.linalg.norm(err))
            if d < tol:
                return True, d
            if d < best - 0.001:
                best = d
                stall = 0
            else:
                stall += 1
                if stall >= 4:
                    log(f"move_to[{label}] stalled d={d:.4f}")
                    return d < 3 * tol, d
            mag = min(vmax, d / POS_SCALE)
            delta = err / d * mag
            k = 1
            if d > 0.03:
                k = int(max(1, min(4, (d - 0.015) / (mag * POS_SCALE))))
            row = [delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip]
            self.step([row] * k)
        d = float(np.linalg.norm(goal - self.pos()))
        log(f"move_to[{label}] max calls d={d:.4f}")
        return d < 3 * tol, d

    def hold(self, grip, n):
        self.step([[0, 0, 0, 0, 0, 0, grip]] * n)

    def rotate_z(self, angle, grip):
        """Rotate the end effector about world z by angle (rad), in <=0.25 rad rows."""
        n = max(1, int(math.ceil(abs(angle) / 0.25)))
        per = angle / n / ROT_SCALE
        self.step([[0, 0, 0, 0, 0, per, grip]] * n + [[0, 0, 0, 0, 0, 0, grip]])

    def yaw(self):
        R = rotvec_to_mat(self.state[3:6])
        x = R[:, 0]
        return math.atan2(x[1], x[0])


# ---------------------------------------------------------------- live perception helpers
def color_from_instruction(text):
    t = (text or "").lower()
    best = None
    for c in ("red", "green", "blue"):
        i = t.find(c + " cube")
        if i < 0:
            i = t.find(c)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, c)
    return best[1] if best else "red"


def mask_samples(mask, n=24, erode=1):
    from scipy import ndimage
    m = ndimage.binary_erosion(mask, iterations=erode) if mask.sum() > 40 else mask
    if m.sum() < 3:
        m = mask
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return [(float(xs[i]), float(ys[i])) for i in idx]


def largest_blob(mask, near=None):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    best = None
    for k in range(1, n + 1):
        m = lab == k
        a = int(m.sum())
        if a < 12:
            continue
        ys, xs = np.nonzero(m)
        score = a
        if near is not None:
            score = a / (1.0 + np.hypot(xs.mean() - near[0], ys.mean() - near[1]) / 30.0)
        if best is None or score > best[0]:
            best = (score, m)
    return None if best is None else best[1]


def top_face_point(pts, tol=0.007):
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None, None
    A = np.array(pts)
    zmax = np.percentile(A[:, 2], 90)
    top = A[A[:, 2] > zmax - tol]
    if len(top) == 0:
        return None, None
    return np.median(top, 0) if len(top) < 4 else top.mean(0), float(zmax)


def locate_cube_front(rb, color):
    m = P.cube_mask(rb.front, color)
    blob = largest_blob(m)
    if blob is None:
        return None
    ys, xs = np.nonzero(blob)
    c = (float(xs.mean()), float(ys.mean()))
    pts = rb.pix2world("front", mask_samples(blob, 20))
    xyz, ztop = top_face_point(pts)
    # table points around the cube
    wood = P.wood_mask(rb.front)
    ring = []
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for rad in (16, 22):
            u = c[0] + rad * math.cos(ang)
            v = c[1] + rad * math.sin(ang) + 4
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < 256 and 0 <= vi < 256 and wood[vi, ui]:
                ring.append((u, v))
    tpts = [p for p in rb.pix2world("front", ring[:16]) if p is not None]
    table_z = float(np.median([p[2] for p in tpts])) if len(tpts) >= 3 else None
    return {"px": c, "xyz": xyz, "ztop": ztop, "table_z": table_z, "n_table": len(tpts),
            "n_pts": sum(p is not None for p in pts)}


def locate_dest(rb, px, homes_area, table_z):
    u0, v0 = px
    rad = max(3.0, 0.5 * math.sqrt(homes_area / math.pi))
    uvs = [(u0, v0)] + [(u0 + rad * math.cos(a), v0 + rad * math.sin(a))
                        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    pts = rb.pix2world("front", uvs)
    good = [p for p in pts if p is not None]
    if not good:
        return None
    A = np.array(good)
    if table_z is not None:
        sel = A[np.abs(A[:, 2] - table_z) < 0.012]
        if len(sel) >= 3:
            A = sel
    if pts[0] is not None and (table_z is None or abs(pts[0][2] - table_z) < 0.012):
        # centre pixel valid on the table: average centre with ring mean
        return 0.5 * (pts[0] + A.mean(0))
    return A.mean(0)


def wrist_cube(rb, color, expect_xy=None):
    if rb.wrist is None:
        return None
    m = P.cube_mask(rb.wrist, color)
    blob = largest_blob(m)
    if blob is None:
        return None
    pts = rb.pix2world("wrist", mask_samples(blob, 24))
    xyz, ztop = top_face_point(pts)
    ang = None
    try:
        import cv2
        ys, xs = np.nonzero(blob)
        rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        a = rect[2] % 90.0
        if a > 45:
            a -= 90.0
        ang = math.radians(a)
    except Exception as e:
        log("minAreaRect failed", repr(e))
    if xyz is not None and expect_xy is not None and np.hypot(*(xyz[:2] - expect_xy[:2])) > 0.06:
        log("wrist cube far from expectation; ignoring", xyz, expect_xy)
        return {"xyz": None, "ztop": None, "angle": ang, "area": int(blob.sum())}
    return {"xyz": xyz, "ztop": ztop, "angle": ang, "area": int(blob.sum())}


# ---------------------------------------------------------------- task steps
def analyse(rb, meta0, color):
    man, full = rio.manifest_frames(meta0)
    if not man:
        log("no initial manifest")
        return None
    frames = sorted(man.get("frames", []), key=lambda f: f.get("frame_index", 0))
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") != "video_demo"]
    log(f"manifest frames={len(frames)} video={len(vid)} exec={len(exe)} keys={list(man.keys())[:12]}")
    seq = vid + exe[-1:]
    ims, ts = [], []
    for f in seq:
        p = rio.frame_file(f, "front", "rgb", full)
        try:
            ims.append(rio.load_rgb(p))
            ts.append(float(f.get("timestamp_s", len(ts) * 0.05)))
        except Exception as e:
            log("frame load failed", p, repr(e))
    if exe:
        # timestamps of the execution frame restart at zero; keep them monotone
        if len(ts) >= 2 and ts[-1] <= ts[-2]:
            ts[-1] = ts[-2] + 0.05
    if len(ims) < 10:
        log("too few frames", len(ims))
        return {"frames": ims, "times": ts, "res": None}
    t = time.time()
    res = P.analyse_video(ims, ts, color, log=lambda *a: log("[perc]", *a))
    log(f"analysis took {time.time() - t:.1f}s ok={res.get('ok')} why={res.get('why')}")
    return {"frames": ims, "times": ts, "res": res}


def pick(rb, color, cube, table_z, cube_h):
    """One pick attempt. Returns grasp TCP height above table, or None."""
    xyz = np.array(cube["xyz"], float)
    ztop = cube["ztop"] if cube.get("ztop") is not None else table_z + cube_h
    hover = np.array([xyz[0], xyz[1], ztop + HOVER_ABOVE_TOP])
    # rise first if below hover height, then translate
    if rb.pos()[2] < hover[2] - 0.02:
        rb.move_to([rb.pos()[0], rb.pos()[1], hover[2]], -1, tol=0.01, label="rise")
    rb.move_to(hover, -1, tol=0.006, label="hover")
    # wrist refinement (+ yaw alignment)
    target_xy = xyz[:2].copy()
    try:
        rb.observe()
        w = wrist_cube(rb, color, expect_xy=xyz)
        log("wrist cube 1:", w)
        if w and w.get("xyz") is not None:
            target_xy = w["xyz"][:2]
            if w.get("ztop") is not None and abs(w["ztop"] - ztop) < 0.02:
                ztop = 0.5 * (ztop + w["ztop"])
        if w and w.get("angle") is not None and abs(w["angle"]) > math.radians(12):
            a0 = w["angle"]
            y0 = rb.yaw()
            rb.rotate_z(a0, -1)
            log(f"yaw rotated by {math.degrees(a0):.1f}; state yaw {math.degrees(y0):.1f}->{math.degrees(rb.yaw()):.1f}")
            rb.observe()
            w2 = wrist_cube(rb, color, expect_xy=xyz)
            log("wrist cube 2:", w2)
            if w2 and w2.get("angle") is not None and abs(w2["angle"]) > abs(a0) * 0.75 and abs(w2["angle"]) > math.radians(12):
                rb.rotate_z(-2 * a0, -1)
                rb.observe()
                w2 = wrist_cube(rb, color, expect_xy=xyz)
                log("wrist cube 3 (reversed):", w2)
            if w2 and w2.get("xyz") is not None:
                target_xy = w2["xyz"][:2]
    except EpisodeOver:
        raise
    except Exception as e:
        log("wrist refine failed", repr(e), traceback.format_exc()[-800:])
    cube_h_eff = max(0.015, min(0.08, ztop - table_z))
    grasp_z = table_z + max(GRASP_MIN_ABOVE_TABLE, cube_h_eff * 0.5)
    log(f"pick: target_xy={np.round(target_xy, 4).tolist()} ztop={ztop:.4f} table={table_z:.4f} grasp_z={grasp_z:.4f}")
    rb.move_to([target_xy[0], target_xy[1], ztop + HOVER_ABOVE_TOP], -1, tol=0.004, vmax=VMAX_SLOW, label="align")
    rb.move_to([target_xy[0], target_xy[1], ztop + 0.03], -1, tol=0.005, vmax=VMAX_FAST, label="pre-grasp")
    ok, d = rb.move_to([target_xy[0], target_xy[1], grasp_z], -1, tol=0.004, vmax=VMAX_SLOW, label="descend")
    log(f"descend ok={ok} d={d:.4f} pos={np.round(rb.pos(), 4).tolist()}")
    rb.hold(1, 8)
    f = rb.fingers()
    log(f"closed: fingers={f:.4f}")
    lift = rb.pos().copy()
    lift[2] = table_z + CARRY_ABOVE_TABLE
    rb.move_to(lift, 1, tol=0.01, label="lift")
    f2 = rb.fingers()
    log(f"after lift fingers={f2:.4f}")
    if f2 < FINGER_EMPTY_SUM:
        log("grasp failed (fingers closed)")
        return None
    return grasp_z - table_z


def place(rb, dest, table_z, grasp_h):
    carry = np.array([dest[0], dest[1], table_z + CARRY_ABOVE_TABLE])
    rb.move_to(carry, 1, tol=0.006, label="carry")
    rb.move_to([dest[0], dest[1], table_z + CARRY_ABOVE_TABLE], 1, tol=0.004, vmax=VMAX_SLOW, label="carry-fine")
    zp = table_z + grasp_h + PLACE_CLEARANCE
    rb.move_to([dest[0], dest[1], zp + 0.04], 1, tol=0.006, label="lower")
    ok, d = rb.move_to([dest[0], dest[1], zp], 1, tol=0.004, vmax=VMAX_SLOW, label="lower-fine")
    log(f"at release ok={ok} d={d:.4f} pos={np.round(rb.pos(), 4).tolist()} fingers={rb.fingers():.4f}")
    rb.hold(-1, 6)
    up = rb.pos().copy()
    up[2] = table_z + CARRY_ABOVE_TABLE
    rb.move_to(up, -1, tol=0.01, label="retreat")


def main(ctx):
    log(f"controller {VERSION} start")
    rb = Robot(ctx)
    try:
        rb.status()
        r0, meta0 = rb.observe()
        log("obs0 keys:", list(meta0.keys())[:30] if isinstance(meta0, dict) else type(meta0))
        log("obs0 meta:", str(meta0)[:2000])
        log("state0:", None if rb.state is None else np.round(rb.state, 4).tolist(), "step_id", rb.step_id)
        instr = rio.find_key(meta0, "instruction") or ""
        color = color_from_instruction(instr if isinstance(instr, str) else str(instr))
        log("instruction:", instr, "-> color", color)
    except Exception as e:
        log("startup failed", repr(e), traceback.format_exc()[-1500:])
        return

    an = None
    try:
        an = analyse(rb, meta0, color)
    except Exception as e:
        log("analysis crashed", repr(e), traceback.format_exc()[-1500:])
    res = an.get("res") if an else None
    if an and an.get("frames"):
        rb.front_ref = an["frames"][-1].astype(int)
        if rb.front is None:
            rb.front = an["frames"][-1]
    # re-observe so that image labelling can use the reference
    try:
        rb.observe()
    except Exception as e:
        log("observe failed", repr(e))

    dest_px, home_area = None, 250.0
    if res and res.get("ok"):
        dest_px = res["final_px"]
        home_area = res["homes"][res["final_home"]][2]
    elif res and res.get("homes"):
        # fallback: last placement home of the cube, else the home nearest the image centre
        homes = np.array(res["homes"])
        k = int(np.argmin(np.hypot(homes[:, 0] - 128, homes[:, 1] - 90)))
        dest_px, home_area = homes[k, :2].tolist(), homes[k, 2]
        log("FALLBACK destination home", k)
    log("destination pixel:", dest_px)
    if dest_px is None:
        log("no destination; giving up")
        return

    attempts = 0
    while attempts < MAX_ATTEMPTS and not rb.ended:
        attempts += 1
        try:
            if rb.state is None:
                rb.observe()
            rb.observe()
            cube = locate_cube_front(rb, color)
            log(f"attempt {attempts}: cube front={cube}")
            if cube is None or cube.get("xyz") is None:
                # lift out of view and look again
                p = rb.pos().copy()
                p[2] += 0.08
                rb.move_to(p, -1, tol=0.01, label="clear-view")
                rb.observe()
                cube = locate_cube_front(rb, color)
                log("cube after clearing view:", cube)
                if cube is None or cube.get("xyz") is None:
                    continue
            # already on the destination?
            if np.hypot(cube["px"][0] - dest_px[0], cube["px"][1] - (dest_px[1] - 3.5)) < 3.5 and attempts > 1:
                log("cube already on destination")
                break
            table_z = cube["table_z"]
            if table_z is None:
                table_z = (cube["ztop"] - DEFAULT_CUBE_H) if cube.get("ztop") is not None else rb.pos()[2] - 0.2
            cube_h = (cube["ztop"] - table_z) if cube.get("ztop") is not None else DEFAULT_CUBE_H
            if not (0.015 < cube_h < 0.08) and cube.get("ztop") is not None:
                log(f"implausible cube height {cube_h:.4f}; using default")
                cube_h = DEFAULT_CUBE_H
                table_z = cube["ztop"] - DEFAULT_CUBE_H
            log(f"table_z={table_z:.4f} cube_h={cube_h:.4f} ee={np.round(rb.pos(), 4).tolist()}")
            dest = locate_dest(rb, dest_px, home_area, table_z)
            log("dest world:", None if dest is None else np.round(dest, 4).tolist())
            if dest is None:
                continue
            gh = pick(rb, color, cube, table_z, cube_h)
            if gh is None:
                rb.hold(-1, 4)
                continue
            place(rb, dest, table_z, gh)
            rb.observe()
            m = P.cube_mask(rb.front, color)
            blob = largest_blob(m)
            if blob is not None:
                ys, xs = np.nonzero(blob)
                err = np.hypot(xs.mean() - dest_px[0], ys.mean() - (dest_px[1] - 3.5))
                log(f"verify: cube px=({xs.mean():.1f},{ys.mean():.1f}) dest=({dest_px[0]:.1f},{dest_px[1]:.1f}) err={err:.1f}px")
                if err < 8.0:
                    log("placement verified")
                    break
            else:
                log("verify: cube not visible")
        except EpisodeOver as e:
            log("episode over:", repr(e))
            break
        except Exception as e:
            log("attempt crashed", repr(e), traceback.format_exc()[-1500:])
            try:
                rb.observe()
            except Exception:
                pass
    log(f"done: steps={rb.steps} wall={rb.wall():.0f}s ended={rb.ended} complete={rb.complete} calls={rb.calls}")
