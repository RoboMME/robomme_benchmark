"""ButtonUnmaskSwap controller: perception, tracking and motion on top of the RAX MCP tools."""
import json
import os
import time
import traceback

import numpy as np

import geom
import percep
import rax_io
from tracker import Tracker

OUT = "/workspace/output"
POS_SCALE = 0.05
ROT_SCALE = 0.5


def _np(x):
    return np.asarray(x, dtype=np.float64)


class EpisodeOver(Exception):
    pass


class Log:
    def __init__(self):
        self.t0 = time.time()
        self.path = None
        try:
            os.makedirs(OUT, exist_ok=True)
            self.path = os.path.join(OUT, "ctrl_log.jsonl")
            self.f = open(self.path, "a", buffering=1)
        except Exception:
            self.f = None
        self.n = 0

    def __call__(self, event, **kw):
        rec = {"t": round(time.time() - self.t0, 2), "ev": event}
        for k, v in kw.items():
            try:
                json.dumps(v)
                rec[k] = v
            except Exception:
                rec[k] = repr(v)[:300]
        line = json.dumps(rec)
        self.n += 1
        try:
            if self.f and self.n < 20000:
                self.f.write(line + "\n")
        except Exception:
            pass
        if event not in ("step", "track"):
            try:
                print(line[:600], flush=True)
            except Exception:
                pass

    def dump(self, name, obj):
        try:
            with open(os.path.join(OUT, name), "w") as f:
                json.dump(obj, f, default=lambda o: repr(o)[:200])
        except Exception:
            pass


class Runner:
    def __init__(self, ctx, log, time_budget=1120.0):
        self.ctx = ctx
        self.log = log
        self.t_start = time.time()
        self.time_budget = time_budget
        self.obs = None
        self.step_count = 0
        self.limit = 480
        self.ended = False
        self.success = None
        self.call_times = []
        self.need_depth_observe = None  # None unknown, True when step replies lack depth
        self.on_obs = None  # callback(obs)
        self.raw_dumped = 0
        self.grip = -1.0
        self.last_step_wall = time.time()

    # ---------------- MCP plumbing
    def call(self, tool, **kw):
        t = time.time()
        r = self.ctx.call(tool, **kw)
        dt = time.time() - t
        if tool == "robomme_step":
            self.call_times.append(dt)
        if self.raw_dumped < 6:
            self.raw_dumped += 1
            try:
                meta = rax_io.first_json(r)
                blocks = [{k: (v if k != "data" else f"<{len(v)} b64>") for k, v in b.items()}
                          for b in (r.get("content", []) or [])]
                for b in blocks:
                    if b.get("type") == "text" and len(b.get("text", "")) > 4000:
                        b["text"] = b["text"][:4000] + "...(trunc)"
                self.log.dump(f"raw_{self.raw_dumped}_{tool}.json", {"blocks": blocks, "meta": meta})
            except Exception:
                pass
        return r

    def elapsed(self):
        return time.time() - self.t_start

    def time_left(self):
        return self.time_budget - self.elapsed()

    def avg_step_time(self):
        if not self.call_times:
            return 0.5
        return float(np.median(self.call_times[-20:]))

    def _ingest(self, obs, source):
        if obs.step_id is not None:
            self.step_count = obs.step_id
        if obs.ended:
            self.ended = True
            self.success = obs.success
        if obs.state is None and self.obs is not None:
            obs.state = self.obs.state
        if not obs.rgb and self.obs is not None:
            obs.rgb = self.obs.rgb
        self.obs = obs
        if self.on_obs is not None:
            try:
                self.on_obs(obs)
            except EpisodeOver:
                raise
            except Exception as e:
                self.log("on_obs_error", err=repr(e)[:300], tb=traceback.format_exc()[-1500:])

    def observe(self):
        r = self.call("robomme_observe")
        obs = rax_io.parse_obs(r)
        self._ingest(obs, "observe")
        return obs

    def status(self):
        try:
            r = self.call("robomme_status")
            meta = rax_io.first_json(r)
            self.log.dump("robomme_status.json", meta)
            return meta
        except Exception as e:
            self.log("status_error", err=repr(e)[:300])
            return {}

    def step(self, rows):
        """Send rows (list of 7-vectors). Returns the new observation."""
        if self.ended:
            raise EpisodeOver()
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        remaining = self.limit - self.step_count
        if remaining <= 0:
            self.ended = True
            raise EpisodeOver()
        rows = rows[:max(1, remaining)]
        r = self.call("robomme_step", actions=rows)
        self.last_step_wall = time.time()
        obs = rax_io.parse_obs(r, need_depth=True)
        if obs.step_id is None:
            obs.step_id = self.step_count + len(rows)
        if (len(obs.depth) < 2 or len(obs.rgb) < 2) and not obs.ended:
            if self.need_depth_observe is None:
                self.log("step_reply_incomplete", depth=list(obs.depth), rgb=list(obs.rgb), notes=obs.notes)
            self.need_depth_observe = True
            try:
                r2 = self.call("robomme_observe")
                o2 = rax_io.parse_obs(r2)
                if o2.step_id is None:
                    o2.step_id = obs.step_id
                o2.ended = o2.ended or obs.ended
                if o2.success is None:
                    o2.success = obs.success
                if o2.state is None:
                    o2.state = obs.state
                obs = o2
            except Exception as e:
                self.log("observe_after_step_error", err=repr(e)[:200])
        self._ingest(obs, "step")
        if self.ended:
            self.log("episode_ended", step=self.step_count, success=self.success)
            raise EpisodeOver()
        return obs

    # ---------------- state helpers
    @property
    def pos(self):
        return _np(self.obs.state[:3])

    @property
    def R(self):
        return geom.rotvec_to_mat(self.obs.state[3:6])

    @property
    def fingers(self):
        return float(self.obs.state[6] + self.obs.state[7])


def make_row(runner, target, R_des, grip, vmax=0.5, gain=1.0, rmax=0.3):
    p = runner.pos
    d = (_np(target) - p) * gain / POS_SCALE
    n = np.linalg.norm(d)
    if n > vmax:
        d = d * (vmax / n)
    row = [0.0] * 7
    row[0:3] = d.tolist()
    if R_des is not None:
        e = geom.mat_to_rotvec(R_des @ runner.R.T) / ROT_SCALE
        ne = np.linalg.norm(e)
        if ne > rmax:
            e = e * (rmax / ne)
        row[3:6] = e.tolist()
    row[6] = float(grip)
    return row
