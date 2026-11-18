"""Robot I/O wrapper (robomme MCP tools) with frame collection and closed-loop motion primitives."""
import json
import math
import time

import numpy as np

import geom
import io_utils as iu

POS_SCALE = 0.05
ROT_SCALE = 0.5


class EpisodeEnded(Exception):
    pass


class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = None
        self.grip = -1.0
        self.frames = []          # dicts: key, t_step, phase, state, files{(cam,kind):(path,mpu)}
        self.frame_keys = set()
        self.last_js = None
        self.max_steps = None
        self.remaining = None
        self.n_calls = 0
        self.step_wall = []
        self.summ_logged = 0

    # ------------------------------------------------------------------ parsing
    def _absorb(self, js, tag):
        st = iu.get_state(js)
        if st is not None:
            self.state = st
        sid = iu.get_step_id(js)
        if sid is not None:
            self.step_id = sid
        e = iu.get_bool(js, "episode_ended")
        if e is not None:
            self.ended = self.ended or e
        for key in ("terminated", "truncated", "done"):
            v = iu.get_bool(js, key)
            if v:
                self.ended = True
        c = iu.get_bool(js, "task_complete")
        if c is None:
            c = iu.get_bool(js, "success")
        if c is not None:
            self.complete = c
        lim = iu.find_key(js, "episode_limits", lambda v: isinstance(v, dict))
        if isinstance(lim, dict):
            for k, v in lim.items():
                if isinstance(v, int) and not isinstance(v, bool):
                    kl = k.lower()
                    if "remain" in kl:
                        self.remaining = v
                    elif "max" in kl or "limit" in kl:
                        self.max_steps = v
        self.last_js = js
        if self.summ_logged < 3:
            self.summ_logged += 1
            try:
                self.log(f"[reply-structure {tag}] " + json.dumps(iu.summarize(js))[:3000])
            except Exception:
                pass
        # frames
        for mp in iu.manifest_paths(js):
            try:
                self._load_manifest(mp)
            except Exception as ex:
                self.log(f"manifest load failed {mp}: {ex!r}")

    def _load_manifest(self, mp):
        m, mdir = iu.load_manifest(mp)
        frames = m.get("frames", []) if isinstance(m, dict) else []
        if isinstance(m, list):
            frames = m
        nph = sum(1 for fr in frames if "phase" not in fr)
        for i, fr in enumerate(frames):
            if "phase" in fr:
                phase = str(fr.get("phase"))
            elif nph > 1 and i < len(frames) - 1 and self.step_id == 0:
                phase = "video_demo"
                if fr.get("timestamp_s") is None:
                    fr = dict(fr, timestamp_s=0.05 * i)
            else:
                phase = "execution"
            if phase not in ("video_demo", "execution"):
                phase = "video_demo" if "video" in phase or "demo" in phase else "execution"
            files = iu.frame_files(fr, mdir)
            if ("front", "depth") not in files or ("front", "rgb") not in files:
                continue
            if phase == "video_demo":
                key = ("v", fr.get("frame_index"), fr.get("source_frame_index"), fr.get("timestamp_s"))
                tval = float(fr.get("timestamp_s", 0.0) or 0.0)
            else:
                key = ("e", int(fr.get("step_id", 0) or 0))
                tval = None
            if key in self.frame_keys:
                continue
            self.frame_keys.add(key)
            self.frames.append({"key": key, "phase": phase, "ts": tval,
                                "step": int(fr.get("step_id", 0) or 0),
                                "frame_index": fr.get("frame_index"),
                                "state": fr.get("state") if iu.is_num_list(fr.get("state"), 8) else None,
                                "files": files})

    # ------------------------------------------------------------------ tools
    def observe(self):
        r = self.ctx.call("robomme_observe")
        self.n_calls += 1
        js = iu.reply_json(r)
        self._absorb(js, "observe")
        return js, r

    def status(self):
        r = self.ctx.call("robomme_status")
        js = iu.reply_json(r)
        self._absorb(js, "status")
        return js

    def pixel_to_position(self, step_id, sels, radius=0):
        r = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id),
                          selections=sels, depth_window_radius=int(radius))
        return iu.reply_json(r)

    def step(self, rows):
        if self.ended:
            raise EpisodeEnded()
        rows = [[float(max(-1.0, min(1.0, x))) for x in row] for row in rows]
        if self.remaining is not None and self.remaining <= 0:
            self.ended = True
            raise EpisodeEnded()
        if self.remaining is not None:
            rows = rows[:max(1, self.remaining)]
        t0 = time.monotonic()
        try:
            r = self.ctx.call("robomme_step", actions=rows)
            self.consec_err = 0
        except Exception as ex:
            self.consec_err = getattr(self, "consec_err", 0) + 1
            self.n_err = getattr(self, "n_err", 0) + 1
            self.log(f"robomme_step error #{self.n_err} (consecutive {self.consec_err}): {ex!r}")
            # do not retry the uncertain action; refresh the measured state instead
            try:
                js = iu.reply_json(self.ctx.call("robomme_status"))
                self._absorb(js, "status-after-error")
            except Exception as ex2:
                self.log(f"status after error failed: {ex2!r}")
            if self.consec_err >= 4 or self.ended:
                raise
            return {}
        self.step_wall.append(time.monotonic() - t0)
        self.n_calls += 1
        js = iu.reply_json(r)
        prev = self.step_id
        self._absorb(js, "step")
        if self.step_id == prev:
            self.step_id = prev + len(rows)  # fallback if no step id in reply
        if self.remaining is not None and iu.find_key(js, "episode_limits") is None:
            self.remaining -= len(rows)
        return js

    # ------------------------------------------------------------------ state helpers
    @property
    def pos(self):
        return np.array(self.state[0:3], dtype=float)

    @property
    def R(self):
        return geom.rotvec_to_mat(self.state[3:6])

    @property
    def finger_sum(self):
        return float(self.state[6] + self.state[7])

    finger_axis = 1

    def closing_yaw(self, R=None):
        R = self.R if R is None else R
        a = R[:, self.finger_axis]
        return math.atan2(a[1], a[0])

    def steps_left(self):
        if self.remaining is not None:
            return self.remaining
        if self.max_steps is not None:
            return self.max_steps - self.step_id
        return 480 - self.step_id

    # ------------------------------------------------------------------ motion
    def hold(self, n, grip=None, on_frames=None):
        g = self.grip if grip is None else grip
        self.grip = g
        done = 0
        while done < n and not self.ended:
            k = min(5, n - done)
            self.step([[0, 0, 0, 0, 0, 0, g]] * k)
            done += k
            if on_frames:
                on_frames()

    def move_to(self, target, R_des=None, grip=None, speed=0.5, tol=0.006, rot_tol=0.04,
                max_steps=80, max_chunk=6, on_frames=None, stall_check=False, label=""):
        """Closed-loop move of TCP to target (world), optionally orientation R_des.
        Returns dict(ok, steps, err, stalled)."""
        g = self.grip if grip is None else grip
        self.grip = g
        target = np.asarray(target, dtype=float)
        used = 0
        stalled = False
        hist = []
        while used < max_steps and not self.ended:
            e = target - self.pos
            dist = float(np.linalg.norm(e))
            if R_des is not None:
                re = geom.mat_to_rotvec(R_des @ self.R.T)
            else:
                re = np.zeros(3)
            rerr = float(np.linalg.norm(re))
            if dist < tol and rerr < rot_tol:
                return {"ok": True, "steps": used, "err": dist, "rerr": rerr, "stalled": False}
            v = min(speed, dist / POS_SCALE)
            if dist > 1e-6:
                nsteps = dist / (max(v, 1e-3) * POS_SCALE)
            else:
                nsteps = 1
            k = int(max(1, min(max_chunk, math.floor(nsteps * 0.6))))
            if dist > 3 * tol:
                dvec = e / dist * v
            else:
                dvec = e / POS_SCALE * 0.9
            rk = max(k, 1)
            rvec = re / ROT_SCALE / rk
            rn = np.linalg.norm(rvec)
            if rn > 0.3:
                rvec *= 0.3 / rn
            row = [dvec[0], dvec[1], dvec[2], rvec[0], rvec[1], rvec[2], g]
            k = min(k, max_steps - used)
            p0 = self.pos.copy()
            self.step([row] * k)
            used += k
            if on_frames:
                on_frames()
            moved = float(np.linalg.norm(self.pos - p0))
            hist.append((k, moved, dist))
            if stall_check and k >= 1 and dist > 2 * tol:
                expect = min(dist, k * v * POS_SCALE)
                if moved < 0.25 * expect and len(hist) >= 2 and hist[-2][1] < 0.35 * min(hist[-2][2], hist[-2][0] * v * POS_SCALE):
                    stalled = True
                    return {"ok": False, "steps": used, "err": dist, "rerr": rerr, "stalled": True}
        e = float(np.linalg.norm(target - self.pos))
        return {"ok": e < tol * 2, "steps": used, "err": e, "rerr": None, "stalled": stalled}

    def set_grip(self, g, n=8, on_frames=None):
        self.grip = g
        self.hold(n, grip=g, on_frames=on_frames)
