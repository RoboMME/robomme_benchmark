"""Robot wrapper: observation parsing, live RGB-D retrieval validated against the
pixel_to_position tool, and closed-loop motion primitives on the official delta rows."""
import json
import math

import numpy as np

import vus_geom as vg
from vus_io import (decode_image_block, find_all_keys, find_key, get_bool, get_instruction,
                    get_state, get_step_id, load_frame_images, load_manifest, load_png, log,
                    log_exc, parse_reply)

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = {}
        self.images = []
        self.state = None
        self.step_id = None
        self.ended = False
        self.complete = None
        self.steps_sent = 0
        self.max_steps = None
        self.R0 = None          # reference (downward) orientation
        self.variant = {}       # camera -> ((variant, swap), err)
        self.n_calls = 0
        self.last_limits = None
        self.consec_errors = 0

    # ------------------------------------------------------------------ replies
    def _absorb(self, reply, src):
        meta, images = parse_reply(reply)
        if meta:
            self.meta = meta
        self.images = images
        st = get_state(meta)
        if st is not None:
            self.state = st
        sid = get_step_id(meta)
        if sid is not None:
            self.step_id = sid
        e = get_bool(meta, "episode_ended")
        if e is not None:
            self.ended = self.ended or e
        tc = get_bool(meta, "task_complete")
        if tc is not None:
            self.complete = tc
        lim = find_key(meta, "episode_limits")
        if lim is not None:
            self.last_limits = lim
            self._parse_limits(lim)
        return meta

    def _parse_limits(self, lim):
        try:
            if isinstance(lim, dict):
                for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "steps_limit"):
                    if isinstance(lim.get(k), (int, float)):
                        self.max_steps = int(lim[k])
                        break
        except Exception:
            pass

    def remaining(self):
        lim = self.last_limits
        if isinstance(lim, dict):
            for k in ("remaining_steps", "steps_remaining", "remaining"):
                v = lim.get(k)
                if isinstance(v, (int, float)):
                    return int(v)
        if self.max_steps is not None and self.step_id is not None:
            return self.max_steps - self.step_id
        return 10 ** 6

    def observe(self):
        self.n_calls += 1
        r = self.ctx.call("robomme_observe")
        return self._absorb(r, "observe")

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            meta, _ = parse_reply(r)
            e = get_bool(meta, "episode_ended")
            if e:
                self.ended = True
            lim = find_key(meta, "episode_limits")
            if lim is not None:
                self.last_limits = lim
                self._parse_limits(lim)
            for k in ("max_steps", "step_limit", "max_episode_steps"):
                v = find_key(meta, k)
                if isinstance(v, int) and self.max_steps is None:
                    self.max_steps = v
            return meta
        except Exception:
            log_exc("status")
            return {}

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in r] for r in rows]
        if not rows:
            return
        rem = self.remaining()
        if rem <= 0:
            self.ended = True
            raise EpisodeEnded()
        rows = rows[:max(1, min(100, rem))]
        self.n_calls += 1
        try:
            r = self.ctx.call("robomme_step", actions=rows)
        except Exception:
            log_exc("robomme_step")
            self.consec_errors += 1
            # do not retry blindly: resynchronise from a fresh observation
            try:
                self.status()
                self.observe()
            except Exception:
                log_exc("resync")
            if self.consec_errors >= 3:
                log("three consecutive step errors: treating the episode as over")
                self.ended = True
            if self.ended:
                raise EpisodeEnded()
            return
        self.consec_errors = 0
        self.steps_sent += len(rows)
        prev_sid = self.step_id
        meta = self._absorb(r, "step")
        if get_step_id(meta) is None and prev_sid is not None:
            self.step_id = prev_sid + len(rows)
        top_state = isinstance(meta.get("state"), list) or (
            isinstance(meta.get("observation"), dict) and isinstance(meta["observation"].get("state"), list))
        if not top_state and not self.ended:
            # state not at the top level: a nested per-step state could be stale -> observe
            try:
                self.observe()
            except Exception:
                log_exc("observe after step")
        if self.ended:
            log("episode ended", step=self.step_id, task_complete=self.complete)
            raise EpisodeEnded()

    # ------------------------------------------------------------------ geometry
    def pos(self):
        return np.array(self.state[:3], dtype=float)

    def R(self):
        return vg.rotvec_to_R(self.state[3:6])

    def width(self):
        return float(self.state[6] + self.state[7])

    @staticmethod
    def finger_yaw(R):
        f = R[:, 1]
        return math.atan2(f[1], f[0])

    def set_reference(self):
        R = self.R()
        # keep the start orientation, but if the tool z-axis is nearly vertical, snap it to straight down
        z = R[:, 2]
        if z[2] < -0.9:
            yaw = self.finger_yaw(R)
            # construct R with z = (0,0,-1), y-axis (finger) at yaw
            y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            zz = np.array([0.0, 0.0, -1.0])
            x = np.cross(y, zz)
            self.R0 = np.stack([x, y, zz], axis=1)
        else:
            self.R0 = R
        log("reference orientation", tool_z=z.round(3).tolist(), snapped=bool(z[2] < -0.9),
            finger_yaw=round(self.finger_yaw(R), 3))

    def desired_R(self, yaw):
        if yaw is None:
            yaw = self.finger_yaw(self.R())
        R0 = self.R0 if self.R0 is not None else self.R()
        return vg.Rz(yaw - self.finger_yaw(R0)) @ R0

    # ------------------------------------------------------------------ motion
    def move_to(self, target, yaw=None, grip=-1.0, tol=0.004, rot_tol=0.03, max_steps=120,
                vmax=0.015, wmax=0.12, label="move", stall_calls=4):
        """Closed-loop Cartesian move. Returns dict(ok, err, steps, why)."""
        target = np.asarray(target, dtype=float)
        start = self.steps_sent
        stall = 0
        prev_err = None
        while True:
            p = self.pos()
            e = target - p
            en = float(np.linalg.norm(e))
            Rd = self.desired_R(yaw)
            er = vg.R_to_rotvec(Rd @ self.R().T)
            ern = float(np.linalg.norm(er))
            used = self.steps_sent - start
            if en < tol and ern < rot_tol:
                return {"ok": True, "err": en, "rot_err": ern, "steps": used, "why": "reached"}
            if used >= max_steps:
                log(f"{label}: timeout", err=round(en, 4), rot_err=round(ern, 3), steps=used)
                return {"ok": False, "err": en, "rot_err": ern, "steps": used, "why": "timeout"}
            if prev_err is not None:
                if prev_err - en < 0.002 and en > tol and ern < rot_tol * 3:
                    stall += 1
                else:
                    stall = 0
                if stall >= stall_calls:
                    log(f"{label}: stalled", err=round(en, 4), steps=used, pos=p.round(4).tolist())
                    return {"ok": False, "err": en, "rot_err": ern, "steps": used, "why": "stall"}
            prev_err = en
            v = min(vmax, max(0.003, 0.5 * en))
            npos = max(1, int(math.ceil(en / v)))
            nrot = max(1, int(math.ceil(ern / wmax)))
            n = max(npos, nrot)
            k = min(n, 5)
            dp = e / n
            dr = er / n
            row = [dp[0] / POS_SCALE, dp[1] / POS_SCALE, dp[2] / POS_SCALE,
                   dr[0] / ROT_SCALE, dr[1] / ROT_SCALE, dr[2] / ROT_SCALE, grip]
            self.step([row] * k)

    def hold(self, n, grip):
        rows = [[0, 0, 0, 0, 0, 0, grip]] * int(n)
        while rows:
            self.step(rows[:50])
            rows = rows[50:]

    def gripper(self, grip, n=12, max_n=36):
        """Command the gripper while holding position until the finger width settles.
        Returns the finger width."""
        p = self.pos()
        sent, prev = 0, None
        while sent < max_n:
            e = p - self.pos()
            row = [e[0] / POS_SCALE * 0.5, e[1] / POS_SCALE * 0.5, e[2] / POS_SCALE * 0.5, 0, 0, 0, grip]
            self.step([row] * 6)
            sent += 6
            w = self.width()
            if sent >= n and prev is not None and abs(w - prev) < 0.0008:
                break
            prev = w
        return self.width()

    # ------------------------------------------------------------------ perception I/O
    def tool_points(self, cam, pix, swap=False):
        """pix: list of (col, row). Returns list of (col, row, xyz or None)."""
        out = []
        sid = self.step_id if self.step_id is not None else 0
        for i in range(0, len(pix), 32):
            chunk = pix[i:i + 32]
            sel = []
            for (c, r) in chunk:
                if swap:
                    sel.append({"camera": cam, "u": float(r), "v": float(c)})
                else:
                    sel.append({"camera": cam, "u": float(c), "v": float(r)})
            try:
                rep = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(sid),
                                    selections=sel, depth_window_radius=0)
            except Exception:
                log_exc("pixel_to_position")
                out.extend((c, r, None) for (c, r) in chunk)
                continue
            meta, _ = parse_reply(rep)
            pts = _extract_points(meta, len(chunk))
            for (c, r), p in zip(chunk, pts):
                out.append((c, r, p))
        return out

    def live_images(self, cams=("front", "wrist")):
        """Candidate rgb/depth arrays per camera from every source in the current reply."""
        cands = {c: {"rgb": [], "depth": []} for c in cams}
        # (a) image attachments
        decoded = []
        for b in self.images:
            d = decode_image_block(b)
            if d is not None:
                decoded.append((b, d))
        rgbs = [(b, d[1]) for b, d in decoded if d[0] == "rgb"]
        deps = [(b, d[1]) for b, d in decoded if d[0] == "depth_raw"]
        mpu = 0.001
        v = find_key(self.meta, "meters_per_unit")
        if isinstance(v, (int, float)) and v > 0:
            mpu = float(v)

        def cam_of(block, idx, n):
            txt = json.dumps({k: v for k, v in block.items() if k != "data"})[:400].lower()
            for c in cams:
                if c in txt:
                    return [c]
            return list(cams)  # unknown: try for every camera
        for i, (b, a) in enumerate(rgbs):
            for c in cam_of(b, i, len(rgbs)):
                cands[c]["rgb"].append(("att%d" % i, a))
        for i, (b, a) in enumerate(deps):
            for c in cam_of(b, i, len(deps)):
                cands[c]["depth"].append(("att%d" % i, a * mpu))
        # (b) frame-sequence manifest, last frame
        man, _ = load_manifest(self.meta)
        if isinstance(man, dict) and man.get("frames"):
            fr = man["frames"][-1]
            imgs = load_frame_images(fr, cams)
            for c in cams:
                if "rgb" in imgs.get(c, {}):
                    cands[c]["rgb"].append(("manifest", imgs[c]["rgb"]))
                if "depth" in imgs.get(c, {}):
                    cands[c]["depth"].append(("manifest", imgs[c]["depth"]))
        # (c) any depth paths in the metadata
        for key in ("path", "depth_path", "file"):
            for pth in find_all_keys(self.meta, key, limit=20):
                if isinstance(pth, str) and pth.endswith(".png"):
                    low = pth.lower()
                    for c in cams:
                        if c in low:
                            r = load_png(pth)
                            if r is None:
                                continue
                            if "depth" in low and r[0] in ("depth_raw", "gray"):
                                cands[c]["depth"].append(("path", r[1].astype(float) * mpu))
                            elif r[0] == "rgb":
                                cands[c]["rgb"].append(("path", r[1]))
        return cands

    def validated_depth(self, cam, cands=None, n_samples=16):
        """Pick the depth candidate and projection variant that agree with the tool.
        Returns (depth, rgb, calib, (variant, swap), err) or None."""
        calib = vg.get_calibration(self.meta, cam)
        if calib is None:
            log(f"no calibration for {cam}")
            return None
        if cands is None:
            cands = self.live_images((cam,))
        deps = cands.get(cam, {}).get("depth", [])
        rgbs = cands.get(cam, {}).get("rgb", [])
        if not deps:
            log(f"no depth candidates for {cam}")
            return None
        H, W = deps[0][1].shape[:2]
        rng = np.random.default_rng(0)
        grid = [(int(c), int(r)) for r in np.linspace(20, H - 21, 5) for c in np.linspace(20, W - 21, 5)]
        rng.shuffle(grid)
        best = None
        samples_raw = self.tool_points(cam, grid[:n_samples])
        good = [(c, r, p) for (c, r, p) in samples_raw if p is not None]
        if len(good) < 4:
            log(f"too few tool samples for {cam}; tool-free plane check", n=len(good))
            for name, dep in deps:
                if dep.ndim != 2:
                    continue
                for var in vg.VARIANTS:
                    if var[3] and calib.get("conv") is None and False:
                        continue
                    sc, zt = vg.plane_score(dep, calib, var)
                    if best is None or sc > best[4] + 0.02:
                        rgb = next((ra for rn, ra in rgbs if rn == name), rgbs[0][1] if rgbs else None)
                        best = (dep, rgb, calib, (var, False), sc, name)
            if best is None or best[4] < 0.2:
                log(f"plane check failed for {cam}", score=None if best is None else best[4])
                return None
            log(f"tool-free variant for {cam}", source=best[5], variant=list(best[3][0]), plane_frac=round(best[4], 3))
            self.variant[cam] = (best[3], 0.009)
            return best[0], best[1], best[2], best[3], 0.009
        for name, dep in deps:
            if dep.ndim != 2:
                continue
            (var, sw), err = vg.fit_variant(dep, calib, good)
            if err is None:
                continue
            if best is None or err < best[4]:
                rgb = rgbs[0][1] if rgbs else None
                for rn, ra in rgbs:
                    if rn == name:
                        rgb = ra
                        break
                best = (dep, rgb, calib, (var, sw), err, name)
        if best is None:
            log(f"depth validation failed for {cam}")
            return None
        log(f"depth validated for {cam}", source=best[5], variant=list(best[3][0]), swap=best[3][1],
            median_err_m=round(best[4], 4))
        self.variant[cam] = (best[3], best[4])
        return best[:5]


def _extract_points(meta, n):
    """Find a list of n per-pixel results and pull a 3-vector (or None) from each."""
    lists = []

    def walk(o, depth=0):
        if depth > 6:
            return
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            lists.append(o)
        if isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, list):
            for v in o:
                walk(v, depth + 1)
    walk(meta)
    lst = None
    for l in lists:
        if len(l) == n:
            lst = l
            break
    if lst is None:
        return [None] * n
    out = []
    for d in lst:
        if d.get("valid") is False:
            out.append(None)
            continue
        p = None
        for k in ("position", "world_position", "point", "xyz", "world", "position_m", "world_point"):
            v = d.get(k)
            if isinstance(v, list) and len(v) == 3:
                p = v
                break
            if isinstance(v, dict) and all(a in v for a in "xyz"):
                p = [v["x"], v["y"], v["z"]]
                break
        if p is None:
            for v in d.values():
                if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                    p = v
                    break
        try:
            out.append(np.array(p, dtype=float) if p is not None else None)
        except Exception:
            out.append(None)
    return out
