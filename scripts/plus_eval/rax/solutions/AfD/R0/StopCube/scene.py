"""Scene understanding over time (front camera): button, target, cube track.

Feed one front RGB per control step with `update(step, rgb)`.  Pure functions of the
images; geometry (3-D) is handled by the controller through the camera model.
"""
import numpy as np

from perception import (TableModel, CubeTracker, static_candidates, button_core, components,
                        comp_mask, dilate, to_hsv)


class Scene:
    def __init__(self, rgb0, log=print):
        self.log = log
        self.table = TableModel(rgb0)
        self.shape = rgb0.shape[:2]
        self.tracker = CubeTracker(self.table, self.shape)
        self.rgb0 = rgb0
        self.button = None        # dict(c=(u,v), core=(u,v), comp)
        self.target = None        # dict(c=(u,v), comp, radius_px)
        self.cube_id_step = None
        self.cube_seed = None
        self.steps = []
        self._target_hist = []
        self._target_stable = 0
        self.cands0, _, _ = static_candidates(rgb0, self.table)
        self._init_button(self.cands0, rgb0)
        self.log("scene: table hue %.1f tol %.1f s_lo %.2f ok %s; %d candidates: %s" % (
            self.table.hue, self.table.hue_tol, self.table.s_lo, self.table.ok, len(self.cands0),
            [(round(c["c"][0]), round(c["c"][1]), c["area"], round(c["lowS"], 2), round(c["sat"], 2)) for c in self.cands0]))

    # ------------------------------------------------------------- statics
    def _init_button(self, cands, rgb):
        best, bs = None, -1
        for c in cands:
            if c["area"] < 15:
                continue
            score = c["lowS"] * np.sqrt(c["area"])
            if c["lowS"] > 0.35 and score > bs:
                best, bs = c, score
        if best is not None:
            core = button_core(rgb, best["comp"])
            self.button = {"c": best["c"], "core": core, "comp": best["comp"], "area": best["area"],
                           "w": best["w"], "h": best["h"]}
            self.tracker.static_mask = dilate(comp_mask(self.shape, best["comp"]), 2)

    def _init_target(self, bg_rgb, step):
        """Pick the static coloured object; reject candidates the cube occupied in the bg samples."""
        cands, _, _ = static_candidates(bg_rgb, self.table)
        tr = self.tracker
        sample_steps = [s for s, _ in tr.samples]
        cube_r = 0.8 * np.sqrt(tr.cube_area or 150.0)
        best, bs = None, -1e9
        for c in cands:
            if self.button is not None and np.hypot(c["c"][0] - self.button["c"][0], c["c"][1] - self.button["c"][1]) < 6:
                continue
            if c["area"] < 15 or c["lowS"] > 0.5:
                continue
            occ = 0
            for s in sample_steps:
                d = tr.track.get(s)
                if d is not None and np.hypot(d[0] - c["c"][0], d[1] - c["c"][1]) < cube_r + 0.5 * c["w"]:
                    occ += 1
            if occ > 0.3 * len(sample_steps):
                continue
            score = np.sqrt(c["area"])
            if score > bs:
                best, bs = c, score
        if best is None:
            return False
        prev = self._target_hist[-1] if self._target_hist else None
        self._target_hist.append(best["c"])
        same = prev is not None and np.hypot(prev[0] - best["c"][0], prev[1] - best["c"][1]) < 2.5
        self._target_stable = self._target_stable + 1 if same else 1
        r = 0.5 * best["w"]
        changed = self.target is None or not same
        self.target = {"c": best["c"], "comp": best["comp"], "radius_px": r, "w": best["w"], "h": best["h"],
                       "area": best["area"], "stable": self._target_stable, "since": step}
        tr.target_zone = dilate(comp_mask(self.shape, best["comp"]), 2)
        if changed:
            self.log("scene: target at (%.1f,%.1f) w %d h %d area %d (step %d)" % (
                best["c"][0], best["c"][1], best["w"], best["h"], best["area"], step))
        return True

    # ------------------------------------------------------------- per step
    def update(self, step, rgb):
        self.steps.append(step)
        bg_changed = self.tracker.add_frame(step, rgb)
        tr = self.tracker
        if self.cube_id_step is None:
            if step >= 3 and len(tr.samples) >= 2:
                self._identify_cube(step, rgb)
        if self.cube_id_step is not None and len(tr.samples) >= 4 and self._target_stable < 4 and step % 2 == 0:
            self._init_target(tr.bg.astype(np.uint8), step)
        if self.cube_id_step is not None:
            self._redetect(step, bg_changed or step - self.cube_id_step < 3)

    def _identify_cube(self, step, rgb):
        """The cube is the non-table object that moved since step 0."""
        H0 = to_hsv(self.rgb0)
        nw0 = self.table.nonwood(self.rgb0, H0)
        nw1 = self.table.nonwood(rgb)
        diff = np.abs(rgb.astype(np.int16) - self.rgb0.astype(np.int16)).sum(-1)
        from perception import robot_mask_from
        rob = dilate(robot_mask_from(nw0) | robot_mask_from(nw1), 4)
        moved = (diff > 60) & (nw0 | nw1) & ~rob & ~self.tracker.static_mask
        comps = [c for c in components(moved, 4)]
        if not comps:
            return
        # current cube pixels: changed and non-table now
        now = moved & nw1
        cn = components(now, 4)
        if not cn:
            return
        c = max(cn, key=lambda c: c["area"])
        if c["area"] < 8:
            return
        self.cube_seed = c["c"]
        self.cube_id_step = step
        self.tracker.cube_area = None
        self.log("scene: cube identified at step %d near (%.1f,%.1f) area %d" % (step, c["c"][0], c["c"][1], c["area"]))

    def _redetect(self, step, bg_changed=True):
        tr = self.tracker
        recent = sorted(k for k in tr.frames if k >= step - (12 if bg_changed else 0))
        for s in recent:
            pred, gate = self._predict(s)
            det = tr.detect(s, pred, gate)
            tr.track[s] = det
        det = tr.track.get(step)
        if det is not None:
            areas = [d[2] for k, d in tr.track.items() if d is not None and k >= step - 20]
            if len(areas) >= 3:
                tr.cube_area = float(np.median(areas))

    def _predict(self, s):
        tr = self.tracker
        prev = [(k, tr.track[k]) for k in sorted(tr.track) if k < s and tr.track[k] is not None and k >= s - 8]
        if len(prev) >= 2:
            (k1, d1), (k2, d2) = prev[-2], prev[-1]
            vel = ((d2[0] - d1[0]) / (k2 - k1), (d2[1] - d1[1]) / (k2 - k1))
            dt = s - k2
            p = (d2[0] + vel[0] * dt, d2[1] + vel[1] * dt)
            return p, 10 + 4 * dt + 1.5 * np.hypot(*vel) * dt
        if len(prev) == 1:
            k2, d2 = prev[-1]
            return (d2[0], d2[1]), 14 + 8 * (s - k2)
        if self.cube_seed is not None:
            return self.cube_seed, 25 + 8 * abs(s - self.cube_id_step)
        return None, None

    def cube_px(self, step):
        return self.tracker.track.get(step)
