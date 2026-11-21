"""Task logic for ButtonUnmaskSwap."""
import time
import traceback

import numpy as np

import geom
import percep
import rax_io
from ctrl import EpisodeOver, Runner, make_row, _np
from tracker import Tracker

TARGET_COLORS_DEFAULT = ["blue", "green", "red"]


class Task:
    def __init__(self, runner, log):
        self.r = runner
        self.log = log
        self.models = {}          # cam -> CamModel at calibration pose
        self.calib_state = None
        self.wrist_mode = "ee"    # 'ee' (rigid to EE), 'pose' (camera_poses), 'home', 'none'
        self.home_pos = None
        self.home_R = None
        self.table_z = None
        self.ztop = None
        self.zshoulder = None
        self.roi = None           # (xmin, xmax, ymin, ymax)
        self.buttons = []
        self.cubes = {}           # color -> list of world xyz
        self.cube_px = {}         # color -> {cam: (cy, cx, area)}
        self.targets = []
        self.target_tracks = {}   # color -> Track
        self.tracker = None
        self.tracking = False
        self.p_mode = False       # fallback: pixel_to_position based measurements
        self.static_run = 0
        self.last_meas_n = 0
        self.presses_done = 0
        self.picked = []
        self.held_track = None
        self.finger_axis_home = None
        self.meas_log_every = 5

    # ------------------------------------------------------------ cameras
    def model_for(self, cam, state):
        m = self.models.get(cam)
        if m is None:
            return None
        if cam == "front":
            return m
        if state is None or self.calib_state is None:
            return None
        if self.wrist_mode == "none":
            return None
        p0, R0 = _np(self.calib_state[:3]), geom.rotvec_to_mat(self.calib_state[3:6])
        p, R = _np(state[:3]), geom.rotvec_to_mat(state[3:6])
        if self.wrist_mode == "home":
            if np.linalg.norm(p - p0) < 0.006 and np.linalg.norm(geom.mat_to_rotvec(R @ R0.T)) < 0.04:
                return m
            return None
        if self.wrist_mode == "pose" and self.r.obs is not None:
            mm = self._pose_model(self.r.obs, "wrist")
            if mm is not None:
                return mm
        return m.transformed(R @ R0.T, p, p0)

    def _pose_model(self, obs, cam):
        try:
            intr = rax_io.find_key(obs.meta, ["camera_intrinsics", "intrinsics"])
            poses = rax_io.find_key(obs.meta, ["camera_poses", "extrinsics", "camera_pose"])
            if not isinstance(intr, dict) or not isinstance(poses, dict):
                return None
            K = geom.parse_K(intr.get(cam) if cam in intr else None)
            pose = geom.parse_pose(poses.get(cam) if cam in poses else None)
            kind = getattr(self, "_pose_kind", {}).get(cam)
            for m in geom.pinhole_from(K, pose):
                if m.kind == kind:
                    return m
        except Exception:
            return None
        return None

    def choose_calib_pixels(self, depth, n=16, cam="front"):
        h, w = depth.shape
        valid = depth > 0.01
        if cam == "wrist":
            pass
        vs, us = np.nonzero(valid[::4, ::4])
        vs, us = vs * 4, us * 4
        if vs.size < n:
            return []
        d = depth[vs, us]
        F = np.stack([us * d, vs * d, d], 1)
        # plane in feature space through the dominant surface (table): F.n = 1
        try:
            nvec, *_ = np.linalg.lstsq(F, np.ones(len(F)), rcond=None)
            res = np.abs(F @ nvec - 1)
            # robust refit on the inliers
            inl = res < np.percentile(res, 60)
            nvec, *_ = np.linalg.lstsq(F[inl], np.ones(inl.sum()), rcond=None)
            res = np.abs(F @ nvec - 1)
        except Exception:
            res = np.random.rand(len(F))
        chosen = []

        def far_enough(v, u, md):
            return all((v - a) ** 2 + (u - b) ** 2 >= md * md for a, b in chosen)
        order = np.argsort(-res)
        for k in order:
            if len(chosen) >= n // 2:
                break
            if res[k] < 0.02:
                break
            if far_enough(vs[k], us[k], 24):
                chosen.append((int(vs[k]), int(us[k])))
        # grid points for the rest
        g = int(np.ceil(np.sqrt(n - len(chosen)))) + 1
        for a in np.linspace(12, h - 12, g):
            for b in np.linspace(12, w - 12, g):
                if len(chosen) >= n:
                    break
                v, u = int(a), int(b)
                if valid[v, u] and far_enough(v, u, 10):
                    chosen.append((v, u))
        return chosen[:n]

    def calibrate(self, obs):
        sel = []
        pix = {}
        for cam in ("front", "wrist"):
            d = obs.depth.get(cam)
            if d is None:
                continue
            pix[cam] = self.choose_calib_pixels(d, 16, cam)
            sel += [{"camera": cam, "u": float(u), "v": float(v)} for (v, u) in pix[cam]]
        if not sel:
            # no depth on our side: grid pixels just to learn the reply format
            for cam in ("front", "wrist"):
                for v in (64, 128, 192):
                    for u in (64, 128, 192):
                        sel.append({"camera": cam, "u": float(u), "v": float(v)})
        sid = obs.step_id if obs.step_id is not None else 0
        try:
            rep = self.r.call("robomme_pixel_to_position", observation_step_id=int(sid), selections=sel[:32],
                              depth_window_radius=0)
        except Exception as e:
            self.log("p2p_error", err=repr(e)[:300])
            try:
                rep = self.r.call("robomme_pixel_to_position", observation_step_id=int(sid), selections=sel[:32])
            except Exception as e2:
                self.log("p2p_error2", err=repr(e2)[:300])
                rep = None
        if rep is None:
            for cam in ("front", "wrist"):
                self._pinhole_fallback(obs, cam)
            self.calib_state = obs.state.copy() if obs.state is not None else None
            if "wrist" in self.models:
                self.wrist_mode = "pose"
                self._pose_kind = getattr(self, "_pose_kind", {})
                self._pose_kind["wrist"] = self.models["wrist"].kind
            self.log("calib_done", cams=list(self.models), kinds={c: m.kind for c, m in self.models.items()},
                     wrist_mode=self.wrist_mode, fallback=True)
            return bool(self.models)
        pts = rax_io.parse_points(rep)
        self.log("calib_points", n=len(pts), valid=sum(p is not None for p in pts),
                 sample=[None if p is None else [round(float(x), 4) for x in p] for p in pts[:4]])
        if not pix:
            self.p_mode = True
            return False
        k = 0
        for cam in ("front", "wrist"):
            if cam not in pix:
                continue
            P = pix[cam]
            seg = pts[k:k + len(P)]
            k += len(P)
            d = obs.depth[cam]
            rows = [(v, u, p) for (v, u), p in zip(P, seg) if p is not None]
            if len(rows) < 8:
                self.log("calib_few", cam=cam, n=len(rows))
                continue
            vs = np.array([a for a, _, _ in rows], float)
            us = np.array([b for _, b, _ in rows], float)
            X = np.array([p for _, _, p in rows])
            best = None
            # hypothesis 1: u=column, v=row
            d1 = d[vs.astype(int), us.astype(int)]
            f1 = geom.fit_linear(us, vs, d1, X)
            if f1 is not None:
                best = ("uv", f1)
            # hypothesis 2: service reads u as row: then pixel queried is (row=u, col=v)
            try:
                d2 = d[us.astype(int), vs.astype(int)]
                f2 = geom.fit_linear(vs, us, d2, X)
                if f2 is not None and (best is None or f2[0].rms < best[1][0].rms * 0.5):
                    best = ("vu", f2)
            except Exception:
                pass
            # pinhole candidates from metadata
            pin = None
            try:
                intr = rax_io.find_key(obs.meta, ["camera_intrinsics", "intrinsics"])
                poses = rax_io.find_key(obs.meta, ["camera_poses", "extrinsics", "camera_pose"])
                K = geom.parse_K(intr.get(cam)) if isinstance(intr, dict) and cam in intr else None
                pose = geom.parse_pose(poses.get(cam)) if isinstance(poses, dict) and cam in poses else None
                for m in geom.pinhole_from(K, pose):
                    pred = m.backproject(us, vs, d1)
                    rms = float(np.sqrt(np.mean(np.sum((pred - X) ** 2, 1))))
                    if pin is None or rms < pin[1]:
                        pin = (m, rms)
            except Exception as e:
                self.log("pinhole_err", err=repr(e)[:200])
            model, res, cond = best[1]
            self.log("calib_fit", cam=cam, hyp=best[0], rms=model.rms, cond=cond,
                     maxres=float(res.max()), pin=None if pin is None else [pin[0].kind, pin[1]],
                     trans=[round(float(x), 4) for x in model.t])
            chosen = None
            if model.rms < 0.006 and cond < 1e7:
                chosen = model
            if pin is not None and pin[1] < 0.006 and (chosen is None or pin[1] < model.rms * 1.5):
                if chosen is None or pin[1] <= model.rms + 0.002:
                    chosen = pin[0]
                    self._pose_kind = getattr(self, "_pose_kind", {})
                    self._pose_kind[cam] = pin[0].kind
            if chosen is None and model.rms < 0.015:
                chosen = model
            if chosen is not None:
                if best[0] == "vu" and chosen is model:
                    self.log("calib_note", msg="service u/v are swapped relative to columns/rows")
                    self.uv_swapped = True
                self.models[cam] = chosen
        for cam in ("front", "wrist"):
            if cam not in self.models and obs.depth.get(cam) is not None:
                self._pinhole_fallback(obs, cam)
                if cam == "wrist" and "wrist" in self.models:
                    self._pose_kind = getattr(self, "_pose_kind", {})
                    self._pose_kind["wrist"] = self.models["wrist"].kind
        self.calib_state = obs.state.copy() if obs.state is not None else None
        if "wrist" in self.models and getattr(self, "_pose_kind", {}).get("wrist"):
            self.wrist_mode = "pose"
        self.log("calib_done", cams=list(self.models), kinds={c: m.kind for c, m in self.models.items()},
                 wrist_mode=self.wrist_mode)
        return bool(self.models)

    def _pinhole_fallback(self, obs, cam):
        """Pick a metadata pinhole convention whose back-projection puts a horizontal table below the camera."""
        try:
            d = obs.depth.get(cam)
            if d is None:
                return None
            intr = rax_io.find_key(obs.meta, ["camera_intrinsics", "intrinsics"])
            poses = rax_io.find_key(obs.meta, ["camera_poses", "extrinsics", "camera_pose"])
            K = geom.parse_K(intr.get(cam)) if isinstance(intr, dict) and cam in intr else None
            pose = geom.parse_pose(poses.get(cam)) if isinstance(poses, dict) and cam in poses else None
            if K is None or pose is None:
                return None
            vs, us = np.mgrid[4:d.shape[0] - 4:6, 4:d.shape[1] - 4:6]
            vs, us = vs.ravel(), us.ravel()
            dd = d[vs, us]
            ok = dd > 0.01
            vs, us, dd = vs[ok], us[ok], dd[ok]
            if vs.size < 50:
                return None
            best = None
            cam_z = float(np.asarray(pose[0])[2])
            for m in geom.pinhole_from(K, pose):
                X = m.backproject(us.astype(float), vs.astype(float), dd)
                z = X[:, 2]
                hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
                if hist.size == 0:
                    continue
                k = int(np.argmax(hist))
                zmode = 0.5 * (edges[k] + edges[k + 1])
                frac = float(np.mean(np.abs(z - zmode) < 0.012))
                score = frac - (1.0 if zmode > cam_z else 0.0)
                if best is None or score > best[0] + 1e-6:
                    best = (score, m, zmode, frac)
            if best is None or best[3] < 0.2:
                self.log("pinhole_fallback_fail", cam=cam, best=None if best is None else best[3])
                return None
            self.models[cam] = best[1]
            self.log("pinhole_fallback", cam=cam, kind=best[1].kind, table_z=round(float(best[2]), 4),
                     frac=round(best[3], 3))
            return best[1]
        except Exception as e:
            self.log("pinhole_fallback_error", cam=cam, err=repr(e)[:200])
            return None

    def verify_wrist(self, obs):
        """Check the wrist model at a new pose against the service; downgrade if wrong."""
        if "wrist" not in self.models or obs.depth.get("wrist") is None or obs.state is None:
            return
        try:
            d = obs.depth["wrist"]
            P = self.choose_calib_pixels(d, 10, "wrist")
            if len(P) < 6:
                return
            sel = [{"camera": "wrist", "u": float(u), "v": float(v)} for (v, u) in P]
            if getattr(self, "uv_swapped", False):
                sel = [{"camera": "wrist", "u": float(v), "v": float(u)} for (v, u) in P]
            rep = self.r.call("robomme_pixel_to_position", observation_step_id=int(obs.step_id or 0),
                              selections=sel, depth_window_radius=0)
            pts = rax_io.parse_points(rep)
            rows = [(v, u, p) for (v, u), p in zip(P, pts) if p is not None]
            if len(rows) < 6:
                return
            vs = np.array([a for a, _, _ in rows], float)
            us = np.array([b for _, b, _ in rows], float)
            X = np.array([p for _, _, p in rows])
            dd = d[vs.astype(int), us.astype(int)]
            out = {}
            for mode in ("ee", "pose"):
                self.wrist_mode, old = mode, self.wrist_mode
                m = self.model_for("wrist", obs.state)
                self.wrist_mode = old
                if m is None:
                    continue
                pred = m.backproject(us, vs, dd)
                out[mode] = float(np.sqrt(np.mean(np.sum((pred - X) ** 2, 1))))
            fit = geom.fit_linear(us, vs, dd, X)
            self.log("verify_wrist", res=out, refit_rms=None if fit is None else fit[0].rms, step=self.r.step_count)
            good = [k for k, v in out.items() if v < 0.012]
            if good:
                self.wrist_mode = min(good, key=lambda k: out[k])
            else:
                self.wrist_mode = "home"
        except Exception as e:
            self.log("verify_wrist_err", err=repr(e)[:300])

    # ------------------------------------------------------------ scene geometry
    def points(self, obs, cam, mask):
        m = self.model_for(cam, obs.state)
        d = obs.depth.get(cam)
        if m is None or d is None or mask is None:
            return np.zeros((0, 3)), np.zeros((0, 2), int)
        return percep.world_points(m, d, mask)

    def pix_world(self, obs, cam, v, u, win=1):
        """World point of one pixel using a median depth window."""
        m = self.model_for(cam, obs.state)
        d = obs.depth.get(cam)
        if m is None or d is None:
            return None
        h, w = d.shape
        v0, v1 = max(0, int(v) - win), min(h, int(v) + win + 1)
        u0, u1 = max(0, int(u) - win), min(w, int(u) + win + 1)
        patch = d[v0:v1, u0:u1]
        patch = patch[patch > 0.01]
        if patch.size == 0:
            return None
        dd = float(np.median(patch))
        return m.backproject(float(u), float(v), dd)

    def estimate_table(self, obs):
        zs = []
        for cam in ("front", "wrist"):
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            hsv = percep.rgb_to_hsv(rgb)
            tab = (hsv[..., 0] >= 4) & (hsv[..., 0] <= 18) & (hsv[..., 1] > 90) & (hsv[..., 1] < 215)
            X, _ = self.points(obs, cam, tab)
            if X.shape[0] > 50:
                zs.append(float(np.median(X[:, 2])))
        if zs:
            self.table_z = float(np.median(zs))
        self.log("table", z=self.table_z, per_cam=zs)

    def detect_cubes(self, obs):
        colors = list(dict.fromkeys(self.targets + ["red", "green", "blue", "yellow", "purple", "orange", "black"]))
        found = {}
        for cam in ("wrist", "front"):
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            hsv = percep.rgb_to_hsv(rgb)
            for c in colors:
                cm = percep.color_mask(hsv, c)
                if cam == "wrist":
                    cm &= ~percep.gripper_region_mask(cm.shape)
                comps = percep.components(cm, 4)
                for area, cy, cx, ys, xs in comps:
                    if area > 900:
                        continue
                    X, _ = self.points(obs, cam, self._mask_from(ys, xs, cm.shape))
                    if X.shape[0] < 2:
                        continue
                    p = np.median(X, 0)
                    found.setdefault(c, []).append({"cam": cam, "area": int(area), "px": (cy, cx), "xyz": p})
        # fuse per colour: cluster detections within 3 cm
        cubes = {}
        for c, dets in found.items():
            clusters = []
            for dct in sorted(dets, key=lambda x: (x["cam"] != "wrist", -x["area"])):
                for cl in clusters:
                    if np.linalg.norm(cl[0]["xyz"][:2] - dct["xyz"][:2]) < 0.03:
                        cl.append(dct)
                        break
                else:
                    clusters.append([dct])
            items = []
            for cl in clusters:
                xyz = np.mean([x["xyz"] for x in cl], 0)
                area = sum(x["area"] for x in cl)
                items.append({"xyz": xyz, "area": area, "dets": cl})
            items.sort(key=lambda x: -x["area"])
            cubes[c] = items
        self.cubes = cubes
        self.log("cubes", summary={c: [[round(float(v), 4) for v in it["xyz"]] + [it["area"]] for it in its]
                                   for c, its in cubes.items()})
        return cubes

    @staticmethod
    def _mask_from(ys, xs, shape):
        m = np.zeros(shape[:2], bool)
        m[ys, xs] = True
        return m

    def detect_buttons(self, obs):
        cands = []
        for cam in ("wrist", "front"):
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            for cy, cx, box, gfrac in percep.detect_buttons_img(rgb):
                # (fingers / hand are rejected by the 3-D height filter below)
                if box > 1500:
                    continue
                p = self.pix_world(obs, cam, cy, cx, win=1)
                if p is None:
                    continue
                cands.append({"cam": cam, "xyz": p, "box": box, "g": gfrac})
        tz = self.table_z if self.table_z is not None else None
        good = []
        for c in cands:
            if tz is not None and not (tz - 0.01 < c["xyz"][2] < tz + 0.08):
                continue
            good.append(c)
        # fuse across cameras
        fused = []
        for c in sorted(good, key=lambda x: x["cam"] != "wrist"):
            for f in fused:
                if np.linalg.norm(f["xyz"][:2] - c["xyz"][:2]) < 0.03:
                    f["n"] += 1
                    f["cams"].append(c["cam"])
                    if c["cam"] == "wrist":
                        f["xyz_w"] = c["xyz"]
                    break
            else:
                fused.append({"xyz": c["xyz"], "n": 1, "cams": [c["cam"]], "box": c["box"]})
        # remove candidates that coincide with cubes
        fused.sort(key=lambda f: (-f["n"], -f["box"]))
        pick = fused[:2]
        if len(fused) > 2:
            # choose the pair most similar in height and 5..35 cm apart, prefer seen by both cams
            best = None
            for i in range(len(fused)):
                for j in range(i + 1, len(fused)):
                    a, b = fused[i], fused[j]
                    dist = np.linalg.norm(a["xyz"][:2] - b["xyz"][:2])
                    if not (0.04 < dist < 0.4):
                        continue
                    score = a["n"] + b["n"] - 20 * abs(a["xyz"][2] - b["xyz"][2])
                    if best is None or score > best[0]:
                        best = (score, [a, b])
            if best:
                pick = best[1]
        if len(pick) >= len(self.buttons):
            self.buttons = [f["xyz"].copy() for f in pick]
        self.log("buttons", cands=len(cands), fused=[[round(float(v), 4) for v in f["xyz"]] + [f["n"]] for f in fused],
                 chosen=[[round(float(v), 4) for v in b] for b in self.buttons])
        return self.buttons

    # ------------------------------------------------------------ container measurements
    def container_mask(self, rgb):
        hsv = percep.rgb_to_hsv(rgb)
        m = percep.whiteish(hsv)
        r = rgb[..., 0].astype(int)
        b = rgb[..., 2].astype(int)
        m &= (b - r) <= 8          # bluish-gray occluder pillar excluded
        return m

    def measure(self, obs, band=None, cams=("wrist", "front")):
        """Container top centres seen in this observation: list of dicts (xy, n, yaw, cam)."""
        if self.ztop is None:
            return []
        lo, hi = band if band is not None else (self.ztop - 0.013, self.ztop + 0.012)
        ee = _np(obs.state[:3]) if obs.state is not None else None
        per_cam = {}
        for cam in cams:
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            mask = self.container_mask(rgb)
            X, pix = self.points(obs, cam, mask)
            if X.shape[0] == 0:
                continue
            keep = (X[:, 2] > lo) & (X[:, 2] < hi)
            if self.roi is not None:
                x0, x1, y0, y1 = self.roi
                keep &= (X[:, 0] > x0) & (X[:, 0] < x1) & (X[:, 1] > y0) & (X[:, 1] < y1)
            if ee is not None:
                dxy = np.linalg.norm(X[:, :2] - ee[:2], axis=1)
                keep &= ~((dxy < 0.075) & (X[:, 2] > ee[2] - 0.03))
                if self.held_track is not None:
                    keep &= ~(dxy < 0.06)
            if cam == "wrist" and obs.depth.get("wrist") is not None:
                dd = obs.depth["wrist"][pix[:, 0], pix[:, 1]]
                keep &= dd > 0.07
            X = X[keep]
            pix = pix[keep]
            H_, W_ = rgb.shape[:2]
            if X.shape[0] < 6:
                per_cam[cam] = []
                continue
            comps = percep.grid_components(X[:, :2], cell=0.004, min_pts=6, dilate=1)
            items = []
            for idx in comps:
                P = X[idx, :2]
                ext = np.ptp(P, axis=0)
                if max(ext) > 0.11:
                    continue
                pr, pc = pix[idx, 0], pix[idx, 1]
                if pr.min() <= 2 or pc.min() <= 2 or pr.max() >= H_ - 3 or pc.max() >= W_ - 3:
                    continue  # truncated at the image border: centre unreliable
                c = np.median(P, 0)
                yaw = percep.square_yaw(P) if cam == "wrist" and idx.size >= 25 else None
                items.append({"xy": c, "n": int(idx.size), "yaw": yaw, "cam": cam,
                              "ext": [float(ext[0]), float(ext[1])], "ztop": float(np.percentile(X[idx, 2], 90))})
            # merge fragments closer than 2 cm
            merged = []
            for it in sorted(items, key=lambda x: -x["n"]):
                for mg in merged:
                    if np.linalg.norm(mg["xy"] - it["xy"]) < 0.02:
                        w = mg["n"] / (mg["n"] + it["n"])
                        mg["xy"] = w * mg["xy"] + (1 - w) * it["xy"]
                        mg["n"] += it["n"]
                        break
                else:
                    merged.append(dict(it))
            per_cam[cam] = [m for m in merged if m["n"] >= 8]
        # fuse cameras: wrist preferred
        out = [dict(m) for m in per_cam.get("wrist", [])]
        for f in per_cam.get("front", []):
            best = None
            for o in out:
                dd = np.linalg.norm(o["xy"] - f["xy"])
                if dd < 0.025 and (best is None or dd < best[0]):
                    best = (dd, o)
            if best is not None:
                o = best[1]
                if o["cam"] == "wrist":
                    o["xy"] = 0.75 * o["xy"] + 0.25 * f["xy"]
                    o["cam"] = "both"
            else:
                out.append(dict(f))
        return out

    def on_obs(self, obs):
        if not self.tracking or self.tracker is None or obs.state is None:
            return
        meas = self.measure(obs)
        self.last_meas_n = len(meas)
        step = int(self.r.step_count)
        diag = self.tracker.update(np.array([m["xy"] for m in meas]) if meas else np.zeros((0, 2)), step,
                                   meas_info=meas)
        if step % self.meas_log_every == 0:
            self.log("track", step=step, n_meas=len(meas), diag=diag, tracks=self.tracker.summary(),
                     targets={c: t.id for c, t in self.target_tracks.items()})

    # ------------------------------------------------------------ motion
    def rows_per_call(self):
        r = self.r
        steps_left = max(1, r.limit - r.step_count)
        tl = r.time_left() - 40
        per = r.avg_step_time() + 0.05
        for k in (1, 2, 3, 4, 5):
            if steps_left * per / k < tl:
                return k
        return 5

    def goto(self, target, R_des, grip, tol=0.004, vmax=0.5, max_steps=40, rot_tol=0.05,
             stall_steps=0, label="", gain=1.0):
        best = 1e9
        stall = 0
        reached = False
        for i in range(max_steps):
            tgt = target() if callable(target) else target
            tgt = _np(tgt)
            err = float(np.linalg.norm(tgt - self.r.pos))
            rerr = 0.0 if R_des is None else float(np.linalg.norm(geom.mat_to_rotvec(R_des @ self.r.R.T)))
            if err < tol and rerr < rot_tol:
                reached = True
                break
            if stall_steps:
                if err < best - 0.0015:
                    best = err
                    stall = 0
                else:
                    stall += 1
                    if stall >= stall_steps:
                        break
            k = self.rows_per_call()
            if err < 0.03:
                k = 1
            row = make_row(self.r, tgt, R_des, grip, vmax=vmax, gain=gain / k)
            self.r.step([row] * k)
        self.log("goto", label=label, reached=reached, err=round(float(np.linalg.norm(_np(tgt) - self.r.pos)), 4),
                 step=self.r.step_count)
        return reached

    def hold(self, n, grip, R_des=None, target=None):
        tgt = self.r.pos.copy() if target is None else _np(target)
        for i in range(n):
            row = make_row(self.r, tgt, R_des, grip, vmax=0.3)
            self.r.step([row])

    def gripper_to(self, grip, R_des=None, max_steps=10, target=None):
        tgt0 = self.r.pos.copy() if target is None else (target if callable(target) else _np(target))
        prev = self.r.fingers
        stable = 0
        for i in range(max_steps):
            tgt = _np(tgt0()) if callable(tgt0) else tgt0
            row = make_row(self.r, tgt, R_des, grip, vmax=0.2)
            self.r.step([row])
            f = self.r.fingers
            if abs(f - prev) < 0.0007 and i >= 2:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = f
        return self.r.fingers

    # ------------------------------------------------------------ phases
    def yaw_R(self, yaw_edge):
        """Desired orientation aligning the finger axis with a container edge direction."""
        if yaw_edge is None or self.finger_axis_home is None:
            return self.home_R
        dyaw = geom.wrap90(yaw_edge - self.finger_axis_home)
        return geom.rotz(dyaw) @ self.home_R

    def setup(self, obs):
        self.home_pos = _np(obs.state[:3]).copy()
        self.home_R = geom.rotvec_to_mat(obs.state[3:6])
        self.calibrate(obs)
        if "wrist" in self.models:
            ax = self.models["wrist"].image_axis_world(0)
            self.finger_axis_home = float(np.arctan2(ax[1], ax[0]))
        Rh = self.home_R
        self.log("home", pos=self.home_pos.tolist(), rotvec=list(map(float, obs.state[3:6])),
                 fingers=float(obs.state[6] + obs.state[7]), finger_axis_wrist=self.finger_axis_home,
                 ee_x=Rh[:, 0].tolist(), ee_y=Rh[:, 1].tolist(), ee_z=Rh[:, 2].tolist())
        self.estimate_table(obs)
        self.detect_cubes(obs)
        self.detect_buttons(obs)
        # region of interest from all cube detections
        allp = [it["xyz"] for its in self.cubes.values() for it in its]
        if allp:
            P = np.array(allp)
            self.roi = (P[:, 0].min() - 0.14, P[:, 0].max() + 0.14, P[:, 1].min() - 0.14, P[:, 1].max() + 0.14)
        self.cube_px = {}
        for c in self.targets:
            if c in self.cubes and self.cubes[c]:
                self.cube_px[c] = self.cubes[c][0]["dets"]
        self.log("setup_done", roi=self.roi, targets=self.targets,
                 target_found={c: (c in self.cubes and bool(self.cubes[c])) for c in self.targets})

    def colour_visible(self, obs, c):
        """Fraction of the initial colour pixels of target c still coloured (static cameras only)."""
        fr = []
        for det in self.cube_px.get(c, []):
            cam = det["cam"]
            if cam == "wrist" and self.model_for("wrist", obs.state) is None and False:
                continue
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            cy, cx = det["px"]
            hsv = percep.rgb_to_hsv(rgb)
            cm = percep.color_mask(hsv, c)
            h, w = cm.shape
            y0, y1 = max(0, int(cy) - 6), min(h, int(cy) + 7)
            x0, x1 = max(0, int(cx) - 6), min(w, int(cx) + 7)
            fr.append(cm[y0:y1, x0:x1].sum() / max(1, det["area"]))
        return max(fr) if fr else None

    def wait_drop(self, max_wait=45):
        """Hold at home until the containers cover the target cubes."""
        obs = self.r.obs
        vis0 = {c: self.colour_visible(obs, c) for c in self.targets}
        self.log("drop_wait_start", vis=vis0, step=self.r.step_count)
        if all(v is None for v in vis0.values()):
            self.log("drop_unknown", msg="no target colours visible at start")
            return False
        for i in range(max_wait):
            obs = self.r.step([make_row(self.r, self.home_pos, self.home_R, -1.0, vmax=0.2)])
            vis_now = [self.colour_visible(obs, c) for c in self.targets]
            vis_now = [v for v in vis_now if v is not None]
            if len(self.buttons) < 2 and i % 3 == 0 and vis_now and min(vis_now) > 0.5:
                try:
                    self.detect_buttons(obs)
                except Exception as e:
                    self.log("button_redetect_error", err=repr(e)[:200])
            vis = {c: self.colour_visible(obs, c) for c in self.targets}
            gone = [c for c, v in vis.items() if v is not None and v < 0.25]
            known = [c for c, v in vis.items() if v is not None]
            if known and len(gone) >= max(1, len(known) - 0) :
                self.log("drop_detected", step=self.r.step_count, vis=vis)
                return True
        self.log("drop_timeout", step=self.r.step_count)
        return False

    def init_tracks(self, obs):
        """Measure container heights and start the tracker at the cube positions."""
        tz = self.table_z if self.table_z is not None else 0.0
        cube_xy = {c: its[0]["xyz"][:2] for c, its in self.cubes.items() if its}
        allcubes = [it["xyz"][:2] for its in self.cubes.values() for it in its]
        zs, zsh = [], []
        for cam in ("wrist", "front"):
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            X, _ = self.points(obs, cam, self.container_mask(rgb))
            if X.shape[0] == 0:
                continue
            for c in allcubes:
                near = np.linalg.norm(X[:, :2] - c, axis=1) < 0.045
                P = X[near & (X[:, 2] > tz + 0.012) & (X[:, 2] < tz + 0.30)]
                if P.shape[0] >= 15:
                    zt = float(np.percentile(P[:, 2], 95))
                    zs.append(zt)
                    if cam == "wrist":
                        lower = P[(P[:, 2] < zt - 0.008) & (P[:, 2] > tz + 0.01), 2]
                        if lower.size >= 10:
                            hist, edges = np.histogram(lower, bins=np.arange(tz + 0.01, zt, 0.003))
                            if hist.size and hist.max() >= 6:
                                k = int(np.argmax(hist))
                                zsh.append(float(0.5 * (edges[k] + edges[k + 1])))
        if zs:
            self.ztop = float(np.median(zs))
        else:
            self.ztop = tz + 0.08
            self.log("ztop_default", ztop=self.ztop)
        if zsh:
            self.zshoulder = float(np.median(zsh))
        self.log("heights", table=tz, ztop=self.ztop, zshoulder=self.zshoulder, n=len(zs),
                 ztops=[round(z, 4) for z in zs][:20])
        meas = self.measure(obs)
        M = [m["xy"] for m in meas]
        self.tracker = Tracker()
        # expected containers: one per cube; seed with measurements, fall back to cube positions
        if len(M) >= max(1, len(allcubes) - 1):
            self.tracker.init(M, int(self.r.step_count))
        else:
            self.log("init_from_cubes", n_meas=len(M), n_cubes=len(allcubes))
            self.tracker.init(allcubes, int(self.r.step_count))
        for c in self.targets:
            if c not in cube_xy:
                continue
            P = np.array([t.pos for t in self.tracker.tracks])
            dd = np.linalg.norm(P - cube_xy[c], axis=1)
            k = int(np.argmin(dd))
            self.target_tracks[c] = self.tracker.tracks[k]
            if dd[k] > 0.035:
                self.log("target_far", color=c, d=float(dd[k]))
        for i, m in enumerate(meas):
            if m.get("yaw") is not None:
                pass
        self.tracking = True
        self.log("tracks_init", n=len(self.tracker.tracks), meas=len(M),
                 targets={c: [t.id, [round(float(v), 4) for v in t.pos]] for c, t in self.target_tracks.items()})

    def find_second_button(self, pressed):
        """Look for another button near the pressed one in the current views."""
        obs = self.r.obs
        best = None
        for cam in ("wrist", "front"):
            rgb = obs.rgb.get(cam)
            if rgb is None:
                continue
            for cy, cx, box, gfrac in percep.detect_buttons_img(rgb):
                p = self.pix_world(obs, cam, cy, cx, win=1)
                if p is None:
                    continue
                d = float(np.linalg.norm(p[:2] - pressed[:2]))
                if not (0.05 < d < 0.35) or abs(p[2] - pressed[2]) > 0.015:
                    continue
                if best is None or d < best[0]:
                    best = (d, p)
        self.log("second_button_search", found=None if best is None else [round(float(v), 4) for v in best[1]])
        return None if best is None else best[1]

    def press_buttons(self):
        if not self.buttons:
            self.log("no_buttons")
            return
        order = sorted(self.buttons, key=lambda b: np.linalg.norm(b[:2] - self.r.pos[:2]))
        if len(order) == 1:
            order = order + [None]
        first = True
        for b in order:
            if b is None:
                b = self.find_second_button(order[0])
                if b is None:
                    # rise for a wider wrist view and retry once
                    self.goto([self.r.pos[0], self.r.pos[1], order[0][2] + 0.2], self.home_R, +1.0, tol=0.01,
                              vmax=0.6, max_steps=10, stall_steps=3, label="btn_search_up")
                    b = self.find_second_button(order[0])
                if b is None:
                    return
                self.buttons.append(b)
            bx, by, bz = [float(v) for v in b]
            if first:
                self.goto([bx, by, bz + 0.07], self.home_R, +1.0, tol=0.008, vmax=0.6, max_steps=30,
                          stall_steps=6, label="btn_above")
                self.verify_wrist(self.r.obs)
                first = False
            else:
                self.goto([bx, by, bz + 0.045], self.home_R, +1.0, tol=0.008, vmax=0.5, max_steps=20,
                          stall_steps=6, label="btn_above2")
            self.goto([bx, by, bz + 0.012], self.home_R, +1.0, tol=0.004, vmax=0.3, max_steps=12,
                      stall_steps=4, label="btn_near")
            # push
            z_before = self.r.pos[2]
            for i in range(7):
                self.r.step([make_row(self.r, [bx, by, bz - 0.018], self.home_R, +1.0, vmax=0.2)])
            z_min = self.r.pos[2]
            self.hold(2, +1.0, self.home_R, target=[bx, by, bz - 0.018])
            self.presses_done += 1
            self.log("pressed", n=self.presses_done, step=self.r.step_count, z_btn=bz, z_before=z_before,
                     z_min=float(z_min), z_now=float(self.r.pos[2]))
            self.goto([bx, by, bz + 0.045], self.home_R, +1.0, tol=0.008, vmax=0.4, max_steps=10,
                      stall_steps=4, label="btn_up")

    def go_home(self, grip=-1.0):
        up = self.home_pos.copy()
        cur = self.r.pos
        if cur[2] < self.home_pos[2] - 0.08:
            self.goto([cur[0], cur[1], min(self.home_pos[2], cur[2] + 0.08)], self.home_R, grip, tol=0.01,
                      vmax=0.6, max_steps=8, stall_steps=3, label="home_up")
        self.goto(up, self.home_R, grip, tol=0.003, vmax=0.6, max_steps=40, rot_tol=0.02, stall_steps=6,
                  label="home")

    def wait_static(self, deadline_step):
        """Hold at home tracking until the shuffle stops or the deadline."""
        run = 0
        n_t = max(1, len(self.targets))
        while self.r.step_count < deadline_step:
            need = 70 * n_t * (self.r.avg_step_time() + 0.15) + 60
            if self.r.time_left() < need:
                self.log("wait_time_limit", step=self.r.step_count, time_left=round(self.r.time_left(), 1))
                return False
            self.r.step([make_row(self.r, self.home_pos, self.home_R, -1.0, vmax=0.2)])
            sp = self.tracker.speeds(window=6)
            n_recent = sum(1 for t in self.tracker.tracks if self.r.step_count - t.last_seen <= 2)
            if sp and max(sp) < 0.0006 and n_recent >= len(self.tracker.tracks) - 1:
                run += 1
            else:
                run = 0
            if self.r.step_count % 10 == 0:
                self.log("wait", step=self.r.step_count, maxspeed=None if not sp else round(float(max(sp)), 5),
                         n_recent=n_recent, run=run)
            if run >= 6:
                self.tracker.static = True
                self.log("static", step=self.r.step_count)
                return True
        self.log("wait_deadline", step=self.r.step_count)
        return False

    def grasp_z(self):
        z = self.ztop - 0.02
        if self.zshoulder is not None and self.ztop - self.zshoulder > 0.012:
            z = 0.5 * (self.ztop + self.zshoulder)
            z = min(max(z, self.ztop - 0.03), self.ztop - 0.01)
        if self.table_z is not None:
            z = max(z, self.table_z + 0.012)
        return z

    def pick(self, color, last):
        T = self.target_tracks.get(color)
        if T is None:
            self.log("pick_no_track", color=color)
            return False
        zg = self.grasp_z()
        hover = self.ztop + 0.09
        low_hover = self.ztop + 0.05
        for attempt in range(8):
            if attempt >= 2 and self.r.limit - self.r.step_count < 25:
                break
            cur = self.r.pos
            if cur[2] < hover - 0.02:
                self.goto([cur[0], cur[1], hover], None, -1.0, tol=0.01, vmax=0.5,
                          max_steps=8, stall_steps=3, label="pick_rise")
            self.goto(lambda: [T.pos[0], T.pos[1], hover], self.yaw_R(T.yaw), -1.0, tol=0.01, vmax=0.6,
                      max_steps=30, rot_tol=0.08, stall_steps=6, label=f"pick_above_{color}")
            self.goto(lambda: [T.pos[0], T.pos[1], low_hover], self.yaw_R(T.yaw), -1.0, tol=0.004, vmax=0.3,
                      max_steps=14, rot_tol=0.05, stall_steps=5, label=f"pick_low_{color}")
            if attempt == 0 and not getattr(self, "_verified_pick", False):
                self._verified_pick = True
                self.verify_wrist(self.r.obs)
                self.hold(1, -1.0, self.yaw_R(T.yaw), target=[T.pos[0], T.pos[1], low_hover])
            xy = T.pos.copy()
            Rg = self.yaw_R(T.yaw)
            self.log("pick_commit", color=color, xy=xy.tolist(), yaw=T.yaw, seen=T.last_seen,
                     step=self.r.step_count, ee=self.r.pos.tolist(), zg=zg)
            moving = not self.tracker.static and float(np.linalg.norm(T.vel)) > 0.0006
            if moving:
                # container still shuffling: follow the (velocity-led) track while descending
                def tgt_fn():
                    p = T.pos + T.vel * 3.0
                    return [p[0], p[1], zg]
            else:
                tgt_fn = [xy[0], xy[1], zg]
            if moving:
                # descend fast and start closing just above grasp height (container keeps moving)
                def tgt_hi():
                    p = tgt_fn()
                    return [p[0], p[1], zg + 0.012]
                self.goto(tgt_hi, Rg, -1.0, tol=0.006, vmax=0.5, max_steps=10, stall_steps=3,
                          label="pick_descend_fast")
                self.goto(tgt_fn, Rg, +1.0, tol=0.004, vmax=0.4, max_steps=3, label="pick_descend_close")
            else:
                self.goto(tgt_fn, Rg, -1.0, tol=0.004, vmax=0.3, max_steps=14, stall_steps=3,
                          label="pick_descend")
            if moving:
                f = self.gripper_to(+1.0, Rg, max_steps=10, target=tgt_fn)
                xy = self.r.pos[:2].copy()
            else:
                f = self.gripper_to(+1.0, Rg, max_steps=10, target=[xy[0], xy[1], zg])
            self.log("grip", color=color, fingers=f, z=float(self.r.pos[2]))
            if f > 0.006:
                break
            self.log("grasp_miss", color=color, attempt=attempt)
            self.gripper_to(-1.0, Rg, max_steps=6)
        else:
            return False
        T.held = True
        self.held_track = T
        lift = self.ztop + 0.14
        self.goto([xy[0], xy[1], lift], Rg, +1.0, tol=0.01, vmax=0.4, max_steps=16, stall_steps=4,
                  label="lift")
        self.log("lifted", color=color, fingers=self.r.fingers, z=float(self.r.pos[2]), step=self.r.step_count)
        self.picked.append(color)
        if last:
            return True
        self.hold(2, +1.0, Rg, target=[xy[0], xy[1], lift])
        # put it back where it was
        self.goto([xy[0], xy[1], zg + 0.006], Rg, +1.0, tol=0.005, vmax=0.35, max_steps=14, stall_steps=3,
                  label="place")
        self.gripper_to(-1.0, Rg, max_steps=7)
        T.held = False
        self.held_track = None
        T.pos = xy.copy()
        T.vel = np.zeros(2)
        T.last_seen = self.r.step_count
        self.goto([xy[0], xy[1], hover], Rg, -1.0, tol=0.01, vmax=0.5, max_steps=10, stall_steps=3,
                  label="retreat")
        return True
