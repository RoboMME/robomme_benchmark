"""v3: highlight memory with cube tracking through the periodic shuffles.

Evidence (memory/PROGRESS.md): cubes shuffle (swap places) about every 100-140 steps for
~30-40 steps; lifting a non-highlighted cube ends the episode as a failure. So every cube
is tracked (front camera always, plus the top-down wrist view at home), the highlight flag
rides on its track, picks run only between shuffles and are aborted before closing the
gripper if a shuffle starts.
"""
import math

import numpy as np

from controller import (CFG, OUT, assign_white, choose_grasp_yaw, detect_cubes, find_button,
                        log, new_white, save_img, wrap, wrist_locate)
from perception import dilate, table_height, white_mask
from robot import EpisodeEnded, MoveAborted, Robot
from tracker import Tracker


class Task3:
    def __init__(self, ctx):
        self.r = Robot(ctx, log)
        self.tz = None
        self.button = None
        self.T = None
        self.in_shuffle = False
        self.shuffle_starts = []
        self.shuffle_ends = []
        self.quiet_cnt = 0
        self.abort_on_shuffle = False
        self.home = None
        self.home_yaw = None
        self.attempts = {}

    # ------------------------------------------------------------ tracking
    def detections(self):
        r = self.r
        bxy = self.button["xy"] if self.button else None
        out = []
        f = r.views.get("front")
        fd = [c["xy"] for c in detect_cubes(f, self.tz, bxy)] if f is not None else []
        at_home = (r.pos[2] > self.home[2] - 0.08 and
                   np.linalg.norm(r.pos[:2] - self.home[:2]) < 0.06)
        if at_home and r.views.get("wrist") is not None:
            wd = [c["xy"] for c in detect_cubes(r.views["wrist"], self.tz, bxy)]
            out = list(wd) + [d for d in fd if all(np.linalg.norm(d - e) > 0.03 for e in wd)]
        else:
            out = fd
        return out

    def track_update(self):
        if self.T is None:
            return
        r = self.r
        try:
            dets = self.detections()
            low = r.pos[2] < 0.25
            moved = self.T.update(dets, r.steps_used,
                                  frozen_xy=r.pos[:2] if low else None, frozen_r=0.07)
        except Exception as e:
            log(f"track update failed {e!r}")
            return
        step = r.steps_used
        if moved >= 3:
            self.quiet_cnt = 0
            if not self.in_shuffle:
                self.in_shuffle = True
                self.shuffle_starts.append(step)
                log(f"SHUFFLE start at step {step} (moved {moved})")
        else:
            if self.in_shuffle:
                self.quiet_cnt += 1
                if step - self.T.last_motion_step >= 6 and self.quiet_cnt >= 3:
                    self.in_shuffle = False
                    self.shuffle_ends.append(step)
                    log(f"SHUFFLE end at step {step}; hl tracks "
                        f"{np.round(self.T.pos[self.T.hl], 3).tolist()}")

    def period(self):
        s = self.shuffle_starts
        if len(s) >= 2:
            return float(np.median(np.diff(s)))
        return None

    def steps_to_next_shuffle(self):
        p = self.period()
        if p is None or not self.shuffle_starts:
            return None
        nxt = self.shuffle_starts[-1] + p
        while nxt < self.r.steps_used - 5:
            nxt += p
        return nxt - self.r.steps_used

    # ------------------------------------------------------------ motions
    def go_home(self, vmax=1.0):
        r = self.r
        r.move_to(self.home, vmax=vmax, tol=0.01, max_steps=60, chunk=3)
        cur = r.finger_yaw()
        if cur is not None and self.home_yaw is not None:
            err = wrap(self.home_yaw - cur)
            if abs(err) > 0.1:
                r.move_to(r.pos, yaw=err, max_steps=12, tol=0.02, chunk=3)

    def wait_for_shuffle_end(self, max_steps=170):
        """Hold at home until a shuffle that is running or starts later has ended (or timeout)."""
        r = self.r
        n0 = len(self.shuffle_ends)
        used = 0
        while used < max_steps:
            if len(self.shuffle_ends) > n0:
                return True
            r.hold(2)
            used += 2
        log(f"wait_for_shuffle_end timeout after {used} steps (in_shuffle={self.in_shuffle})")
        return False

    def hover(self):
        r = self.r
        self.abort_on_shuffle = False
        r.grip = -1.0
        xy = r.pos[:2]
        try:
            todo = self.T.hl & ~self.T.picked
            if todo.any():
                xy = np.mean(self.T.pos[todo], axis=0)
        except Exception:
            pass
        r.move_to([xy[0], xy[1], self.tz + 0.30], vmax=1.0, tol=0.015, max_steps=30, chunk=3)

    def wait_quiet(self, max_steps=80):
        used = 0
        while self.in_shuffle and used < max_steps:
            self.r.hold(2)
            used += 2

    def rotate_to(self, yaw_abs, tries=3):
        r = self.r
        for _ in range(tries):
            cur = r.finger_yaw()
            if cur is None:
                return
            err = wrap(yaw_abs - cur)
            if abs(err) < 0.04:
                return
            r.move_to(r.pos, yaw=err, max_steps=20, tol=0.012, chunk=3)

    def press_button(self, label):
        r = self.r
        b = self.button
        log(f"press_button[{label}] at {np.round(b['xy'], 3).tolist()} top {b['top']:.3f}")
        r.grip = 1.0
        x, y = b["xy"]
        if not b.get("refined"):
            r.move_to([x, y, self.tz + 0.20], vmax=1.0, tol=0.01, max_steps=60, chunk=3)
            r.hold(1)
            try:
                w = r.views["wrist"]
                W = w.world().reshape(-1, 3)
                W = W[np.isfinite(W[:, 2])]
                d = np.linalg.norm(W[:, :2] - b["xy"], axis=1)
                loc = W[(d < 0.028) & (W[:, 2] > self.tz + 0.015) & (W[:, 2] < self.tz + 0.09)]
                if len(loc) > 20:
                    ztop = float(np.percentile(loc[:, 2], 98))
                    cap = loc[loc[:, 2] > ztop - 0.004]
                    nxy = np.median(cap[:, :2], axis=0)
                    if np.linalg.norm(nxy - b["xy"]) < 0.03 and len(cap) >= 5 and ztop < self.tz + 0.05:
                        log(f"button refined -> {np.round(nxy, 4).tolist()} top {ztop:.4f}")
                        b["xy"], b["top"], b["refined"] = nxy, ztop, True
                        x, y = nxy
            except Exception as e:
                log(f"button refine failed {e!r}")
        top = b["top"]
        r.move_to([x, y, top + 0.05], vmax=0.9, tol=0.006, max_steps=50, chunk=3)
        r.hold(3, grip=1.0)
        r.move_to([x, y, top - CFG["press_depth"]], vmax=0.2, tol=0.003, max_steps=30,
                  stall_steps=5, chunk=3)
        log(f"press[{label}] reached z {r.pos[2]:.4f} (top {top:.4f}) at step {r.steps_used}")
        r.hold(2)
        r.move_to([x, y, top + 0.08], vmax=0.8, tol=0.01, max_steps=30, chunk=3)
        r.grip = -1.0

    def abort_check(self):
        return self.abort_on_shuffle and self.in_shuffle

    def pick(self, k):
        """Returns 'done', 'retry' or 'fail'."""
        r = self.r
        T = self.T
        self.attempts[k] = self.attempts.get(k, 0) + 1
        r.grip = -1.0
        self.abort_on_shuffle = True
        try:
            # Approach high over the tracked position, following it.
            for _ in range(3):
                p = T.pos[k].copy()
                r.move_to([p[0], p[1], self.tz + 0.17], vmax=1.0, tol=0.01, max_steps=40, chunk=3)
                if np.linalg.norm(T.pos[k] - p) < 0.01:
                    break
            r.hold(1)
            w = r.views.get("wrist")
            loc = wrist_locate(w, self.tz, T.pos[k], max_dist=0.03) if w is not None else None
            if loc is None:
                log(f"pick {k}: wrist locate failed near {np.round(T.pos[k], 3).tolist()}")
                return "retry"
            save_img(w.rgb, f"scan_{k}_{self.attempts[k]}.png")
            x, y = loc["xy"]
            cube_h = min(0.05, max(0.015, loc["top"] - self.tz))
            yaw, blocked = choose_grasp_yaw(loc, r.finger_yaw())
            log(f"pick {k} att {self.attempts[k]}: track {np.round(T.pos[k], 3).tolist()} -> "
                f"wrist {np.round(loc['xy'], 3).tolist()} h {cube_h:.3f} yaw {math.degrees(yaw):.0f} "
                f"blocked {blocked} step {r.steps_used}")
            cur = r.finger_yaw()
            yerr = wrap(yaw - cur) if cur is not None else 0.0
            r.move_to([x, y, self.tz + 0.075], vmax=0.9, tol=0.005, max_steps=30, chunk=3, yaw=yerr)
            self.rotate_to(yaw)
            # Re-check with a closer wrist view.
            w2 = r.views.get("wrist")
            loc2 = wrist_locate(w2, self.tz, np.array([x, y]), max_dist=0.02) if w2 is not None else None
            if loc2 is not None and np.linalg.norm(loc2["xy"] - loc["xy"]) < 0.015:
                x, y = loc2["xy"]
            gz = self.tz + max(0.010, cube_h * 0.5)
            r.move_to([x, y, gz + 0.02], vmax=0.6, tol=0.004, max_steps=20, chunk=3)
            r.move_to([x, y, gz], vmax=0.3, tol=0.003, max_steps=15, stall_steps=5, chunk=3)
            if self.in_shuffle:
                raise MoveAborted()
            self.abort_on_shuffle = False
        except MoveAborted:
            self.abort_on_shuffle = False
            log(f"pick {k}: aborted (shuffle) at step {r.steps_used}")
            r.move_to([r.pos[0], r.pos[1], self.tz + 0.15], vmax=1.0, tol=0.01, max_steps=10, chunk=3)
            return "retry"
        # Committed: close and lift.
        r.set_gripper(1.0, n=8)
        r.move_to([x, y, gz + 0.08], vmax=0.6, tol=0.006, max_steps=20, chunk=3)
        r.hold(3)
        held = r.opening > 0.008
        log(f"pick {k}: opening {r.opening:.4f} held={held} at step {r.steps_used}")
        save_img(r.views["front"].rgb, f"liftfront_{k}_{self.attempts[k]}.png")
        if held:
            r.move_to([x, y, gz + 0.008], vmax=0.5, tol=0.004, max_steps=20, stall_steps=5, chunk=3)
        r.set_gripper(-1.0, n=6)
        r.move_to([x, y, self.tz + 0.12], vmax=0.9, tol=0.01, max_steps=15, chunk=3)
        T.pos[k] = np.array([x, y])
        return "done" if held else "retry"

    # ------------------------------------------------------------ main
    def run(self):
        r = self.r
        r.observe()
        log(f"instruction: {r.meta.get('instruction')}")
        log(f"state0: {np.round(r.state, 4).tolist()} remaining {r.remaining}")
        self.home = r.pos
        self.home_yaw = r.finger_yaw()
        front, wrist = r.views["front"], r.views["wrist"]
        save_img(front.rgb, "front0.png")
        self.tz = table_height(front)
        self.button = find_button(front, self.tz)
        log(f"table z {self.tz:.4f} button {None if self.button is None else np.round(self.button['xy'], 3).tolist()}")
        self.T = Tracker(self.detections())
        log(f"tracks: {len(self.T.pos)} {np.round(self.T.pos, 3).tolist()}")
        r.after_step = self.track_update
        r.abort_check = self.abort_check

        # 1. Highlight watch at home.
        refw = dilate(white_mask(wrist.rgb, lo=170, spread=45), 2)
        reff = dilate(white_mask(front.rgb, lo=170, spread=45), 2)
        cubes = [dict(xy=p.copy()) for p in self.T.pos]
        acc_w = np.zeros(len(cubes))
        acc_f = np.zeros(len(cubes))
        seen, quiet, used, counts = False, 0, 0, []
        while used < CFG["watch_max_steps"]:
            r.hold(3, grip=-1.0)
            used += 3
            vw, vf = r.views["wrist"], r.views["front"]
            mw, mf = new_white(vw, refw, self.tz), new_white(vf, reff, self.tz)
            n = int(mw.sum()) + int(mf.sum())
            counts.append(n)
            if n >= 20 and not self.in_shuffle:
                seen = True
                quiet = 0
                cubes = [dict(xy=p.copy()) for p in self.T.pos]
                acc_w += assign_white(vw, mw, cubes)
                acc_f += assign_white(vf, mf, cubes)
                if n >= max(counts):
                    save_img(vw.rgb, "highlight_wrist.png")
            elif seen:
                quiet += 1
                if quiet >= 2:
                    break
        sw = acc_w / max(acc_w.max(), 1)
        sf = acc_f / max(acc_f.max(), 1)
        nsrc = int(acc_w.max() > 0) + int(acc_f.max() > 0)
        score = (sw + sf) / max(nsrc, 1)
        self.T.hl = score >= 0.3
        log(f"highlight counts {counts}")
        log(f"scores {np.round(score, 2).tolist()} -> {int(self.T.hl.sum())} highlighted: "
            f"{np.round(self.T.pos[self.T.hl], 3).tolist()}")

        # 2. Wait at home through the first shuffle.
        self.wait_for_shuffle_end(max_steps=150)

        # 3. Start press.
        if self.button is not None:
            self.press_button("start")
        self.hover()
        if self.in_shuffle:
            self.wait_quiet()

        # 4. Picks.
        while True:
            todo = [k for k in np.nonzero(self.T.hl & ~self.T.picked)[0]
                    if self.attempts.get(k, 0) < 4]
            if not todo:
                break
            if r.remaining is not None and r.remaining < 220:
                log("step budget low; stopping picks")
                break
            # Start picks only just after a shuffle has ended (longest quiet window).
            if self.in_shuffle:
                self.wait_quiet(max_steps=120)
            elif self.shuffle_ends and r.steps_used - self.shuffle_ends[-1] > CFG.get("pick_window", 12):
                log(f"{r.steps_used - self.shuffle_ends[-1]} steps since shuffle end; waiting for the next")
                self.wait_for_shuffle_end(max_steps=170)
            cur = r.pos[:2]
            k = min(todo, key=lambda j: np.linalg.norm(self.T.pos[j] - cur))
            try:
                res = self.pick(k)
            except MoveAborted:
                res = "retry"
            finally:
                self.abort_on_shuffle = False
            if res == "done":
                self.T.picked[k] = True
            log(f"pick track {k}: {res}; step {r.steps_used} remaining {r.remaining}; "
                f"period {self.period()} starts {self.shuffle_starts}")
            self.hover()
            if self.in_shuffle:
                self.wait_quiet()
        log(f"picked {int(self.T.picked.sum())}/{int(self.T.hl.sum())}")

        # 5. Stop press.
        if self.button is not None:
            self.press_button("stop")
        r.hold(6)
        log(f"final: ended={r.ended} complete={r.complete} steps={r.steps_used}")
