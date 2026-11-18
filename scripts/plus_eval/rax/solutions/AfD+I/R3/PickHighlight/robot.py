"""Thin robot wrapper: observe / step / closed-loop moves on the official 7-D rows."""
import json
import math
import time

import numpy as np

from perception import parse_reply

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeEnded(Exception):
    pass


class MoveAborted(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.meta = None
        self.views = {}
        self.state = None
        self.grip = -1.0
        self.ended = False
        self.complete = None
        self.steps_used = 0
        self.remaining = None
        self.t_calls = 0.0
        self.n_calls = 0
        self.after_step = None
        self.abort_check = None

    # ------------------------------------------------------------ io
    def _absorb(self, reply):
        meta, views = parse_reply(reply)
        self.meta = meta
        if views:
            self.views = views
        if meta.get("state") is not None:
            self.state = np.asarray(meta["state"], dtype=float)
        lim = meta.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if meta.get("step_id") is not None:
            self.steps_used = meta["step_id"]
        tr = meta.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("episode_ended"):
                self.ended = True
            if s.get("task_complete") is not None:
                if s.get("task_complete"):
                    self.complete = True
                elif self.complete is None and s.get("episode_ended"):
                    self.complete = False
        if meta.get("episode_ended"):
            self.ended = True
        return meta

    def observe(self):
        t = time.time()
        r = self.ctx.call("robomme_observe")
        self.t_calls += time.time() - t
        self.n_calls += 1
        return self._absorb(r)

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, v))) for v in row] for row in rows]
        if self.remaining is not None:
            if self.remaining <= 0:
                raise EpisodeEnded()
            rows = rows[: max(1, int(self.remaining))]
        t = time.time()
        r = self.ctx.call("robomme_step", actions=rows)
        self.t_calls += time.time() - t
        self.n_calls += 1
        meta = self._absorb(r)
        if self.ended:
            self.log(f"EPISODE ENDED at step {self.steps_used} complete={self.complete}")
            raise EpisodeEnded()
        if self.after_step is not None:
            self.after_step()
        return meta

    # ------------------------------------------------------------ helpers
    @property
    def pos(self):
        return self.state[:3].copy()

    @property
    def opening(self):
        return float(self.state[6] + self.state[7])

    def finger_yaw(self):
        """World yaw (rad) of the finger closing axis, taken as the wrist camera's +x axis."""
        w = self.views.get("wrist")
        if w is None:
            return None
        a = w.cam_x_world()
        return math.atan2(a[1], a[0])

    def hold(self, n, grip=None):
        if grip is not None:
            self.grip = grip
        done = 0
        while done < n:
            k = min(50, n - done)
            self.step([[0, 0, 0, 0, 0, 0, self.grip]] * k)
            done += k

    def set_gripper(self, g, n=8):
        self.grip = g
        self.hold(n)

    def move_to(self, target, vmax=0.6, tol=0.004, max_steps=120, chunk=10, yaw=0.0,
                ytol=0.02, stall_steps=12):
        """Closed-loop Cartesian move; `yaw` is an extra world-z rotation (rad) to apply."""
        target = np.asarray(target, dtype=float)
        used = 0
        yaw_left = float(yaw)
        best = 1e9
        since_best = 0
        while used < max_steps:
            if self.abort_check is not None and self.abort_check():
                raise MoveAborted()
            d = target - self.pos
            dist = float(np.linalg.norm(d))
            if dist < tol and abs(yaw_left) < ytol:
                return True
            if dist < best - 0.001:
                best = dist
                since_best = 0
            else:
                since_best += 1
            if since_best > stall_steps and abs(yaw_left) < ytol:
                self.log(f"move_to stall at dist {dist:.4f}")
                return False
            step_len = vmax * POS_SCALE
            n_pos = math.ceil(np.max(np.abs(d)) / step_len) if dist >= tol else 0
            n_rot = math.ceil(abs(yaw_left) / (0.6 * ROT_SCALE)) if abs(yaw_left) >= ytol else 0
            n = max(1, min(chunk, max(n_pos, n_rot), max_steps - used))
            # Position: cover the remaining displacement over max(n_pos, n) rows.
            per = d / max(n_pos, 1) if n_pos > 0 else np.zeros(3)
            if n_pos > 0 and n_pos < n:
                per = d / n
            rows = []
            yaw_rows = 0.0
            for i in range(n):
                p = per if (n_pos > 0) else np.zeros(3)
                if n_rot > 0:
                    dz = yaw_left / max(n_rot, 1)
                else:
                    dz = 0.0
                yaw_rows += dz
                rows.append([p[0] / POS_SCALE, p[1] / POS_SCALE, p[2] / POS_SCALE,
                             0.0, 0.0, dz / ROT_SCALE, self.grip])
            self.step(rows)
            yaw_left -= yaw_rows
            used += n
        return float(np.linalg.norm(target - self.pos)) < tol * 3
