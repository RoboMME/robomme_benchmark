"""StopCube controller: press the button just as the cube reaches the target for the N-th time.

Pipeline (each phase tracks the cube on every control step):
  INIT      observe, parse N, 4 zero steps, identify button (static, tallest/whitest) and cube (moving)
  APPROACH  P-control TCP to (button xy, cap_z + Z_ABOVE), gripper closed
  REFINE    wrist-camera localisation of the cap -> refine xy and cap z (skipped if implausible)
  CALIB     free-air descent test with dz=-1 rows from rest -> descent profile (skipped if time short)
  HOVER     P-control to cap_z + H_HOVER, hold
  WAIT      count arrivals (cube-target excursions), predict N-th closest approach (lap replay /
            quadratic), start PRESS when now + k_reg >= t_N - LEAD - 0.5
  PRESS     dz=-1 rows until episode ends (or PRESS_MAX steps), then hold
Exits: every phase has a step timeout; an urgent N-th arrival forces PRESS from HOVER/CALIB/REFINE
when the TCP is above the button. main() never raises.
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import perception as pc  # noqa: E402
import io_utils as iu  # noqa: E402

VERSION = "stopcube-v1"

# ---------------------------------------------------------------- tunables (offline-derived)
LEAD = 2.0            # press registration this many steps before closest approach (demo leads 0.9/1.8/3.45)
H_HOVER = 0.030       # TCP above cap top while waiting (m)
REG_DEPTH = 0.008     # prior: TCP below cap top at registration (m)
Z_ABOVE = 0.12        # approach height above cap top (m)
TRAVEL_MAX = 0.5      # max action magnitude in travel (x0.05 m)
GRIP = 1.0            # gripper closed
PRESS_MAX = 10        # max press rows
DEFAULT_N = 5
EXPERT_W_REG = 98.0   # wrist white-square width (px) at registration in demos (96/102/103)
OUT_DIR = "/workspace/output"
WALL_LIMIT = 1150.0   # s, controller budget guard (limit 1200)


def now():
    return time.time()


class Log:
    def __init__(self):
        self.t0 = now()
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            self.f = open(os.path.join(OUT_DIR, "stopcube_log.jsonl"), "a")
        except Exception:
            self.f = None

    def __call__(self, kind, **kw):
        rec = {"t": round(now() - self.t0, 2), "kind": kind}
        rec.update(kw)
        try:
            line = json.dumps(rec, default=_jd)
        except Exception:
            line = str(rec)
        try:
            print(line, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(line + "\n")
                self.f.flush()
            except Exception:
                pass


def _jd(o):
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in o.ravel()]
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def save_png(arr, name):
    try:
        from PIL import Image
        Image.fromarray(np.asarray(arr, np.uint8)).save(os.path.join(OUT_DIR, name))
    except Exception:
        pass


# ======================================================================= robot I/O
class Robot:
    def __init__(self, ctx, log):
        self.ctx = ctx
        self.log = log
        self.t = 0                 # executed control steps (observation index)
        self.obs = None
        self.ended = False
        self.complete = None
        self.call_times = []
        self.swap = False          # front/wrist swapped relative to labels
        self.n_logged_raw = 0

    def _raw_log(self, tag, result):
        if self.n_logged_raw >= 4:
            return
        self.n_logged_raw += 1
        try:
            blocks = []
            for b in (result.get("content") or []):
                if b.get("type") == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")[:2500]})
                else:
                    blocks.append({k: (v if k != "data" else "<%d chars>" % len(str(v))) for k, v in b.items()})
            self.log("raw_reply", tag=tag, blocks=blocks,
                     keys=[k for k in result.keys() if k != "content"])
        except Exception as e:
            self.log("raw_reply_err", err=str(e))

    def _absorb(self, o, rows_sent=0):
        if self.swap and o.front is not None and o.wrist is not None:
            o.front, o.wrist = o.wrist, o.front
        prev_t = self.t
        if isinstance(o.step_id, int):
            self.t = o.step_id
        else:
            self.t = prev_t + rows_sent
        if o.ended:
            self.ended = True
        if o.complete is not None:
            self.complete = o.complete
        if o.state is None and self.obs is not None:
            o.state = self.obs.state
        self.obs = o
        return o

    def observe(self):
        t0 = now()
        r = self.ctx.call("robomme_observe")
        self.call_times.append(now() - t0)
        self._raw_log("observe", r)
        o = iu.parse_reply(r)
        o._raw = r
        return self._absorb(o)

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            self._raw_log("status", r)
            o = iu.parse_reply(r)
            return o
        except Exception as e:
            self.log("status_err", err=str(e)[:300])
            return None

    def step(self, rows):
        """Send rows; returns Obs (with front image ensured if possible) or None on error."""
        if self.ended:
            return None
        rows = [[float(np.clip(x, -1.0, 1.0)) for x in r] for r in rows]
        t0 = now()
        try:
            r = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            self.call_times.append(now() - t0)
            self.log("step_err", err=str(e)[:500], t=self.t)
            # do not retry the motion; resync from a fresh observation
            try:
                o = self.observe()
                st = self.status()
                if st is not None and st.ended:
                    self.ended = True
                return o
            except Exception as e2:
                self.log("resync_err", err=str(e2)[:300])
                return None
        self.call_times.append(now() - t0)
        self._raw_log("step", r)
        o = iu.parse_reply(r)
        o._raw = r
        o = self._absorb(o, rows_sent=len(rows))
        if (o.front is None or o.state is None) and not self.ended:
            try:
                o2 = self.observe()
                if o2.front is not None:
                    o = o2
            except Exception as e:
                self.log("observe_err", err=str(e)[:300])
        return o

    def p2p(self, camera, pts):
        """World points for pixel list [(u, v)] of the current observation; list of arrays/None."""
        if not pts:
            return []
        sid = self.obs.step_id if (self.obs is not None and isinstance(self.obs.step_id, int)) else self.t
        sel = [{"camera": camera, "u": float(u), "v": float(v)} for u, v in pts[:32]]
        try:
            r = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(sid),
                              selections=sel, depth_window_radius=1)
        except Exception as e:
            self.log("p2p_err", err=str(e)[:400])
            return [None] * len(pts)
        if self.n_logged_raw < 6:
            self.n_logged_raw += 1
            try:
                txt = [b.get("text", "")[:2000] for b in (r.get("content") or []) if b.get("type") == "text"]
                self.log("raw_p2p", text=txt)
            except Exception:
                pass
        out, _ = iu.parse_points(r, len(sel))
        return out

    def mean_call(self):
        c = self.call_times[-20:]
        return float(np.mean(c)) if c else 0.5


# ======================================================================= controller
class Controller:
    def __init__(self, ctx):
        self.log = Log()
        self.rb = Robot(ctx, self.log)
        self.tr = None
        self.N = DEFAULT_N
        self.button = None          # np.array world (x, y, cap_z)
        self.button_px = None
        self.target_px = None
        self.target_fixed = False
        self.static = []
        self.top_row = 12
        self.profile = None         # list of descent (m) after k press rows from rest
        self.z_reg = None
        self.widths = []            # (tcp_z, width_px) wrist measurements
        self.phase = "INIT"
        self.last_bg = -100
        self.pressed_at = None
        self.decision = {}
        self.wall0 = now()

    # ------------------------------------------------------------ helpers
    def ee(self):
        o = self.rb.obs
        if o is None or o.state is None:
            return None
        return np.asarray(o.state[:3], float)

    def ingest(self, o):
        """Add the newest front frame to the tracker (pad skipped steps with None)."""
        if self.tr is None:
            return
        t = self.rb.t
        while len(self.tr.frames) < t:
            self.tr.add(None)
            self.tr.pos.append(None)
            self.tr.areas.append(None)
        if len(self.tr.frames) == t:
            self.tr.add(None if o is None else o.front)
            if self.tr.bg is not None:
                self.tr.step()
            else:
                self.tr.pos.append(None)
                self.tr.areas.append(None)
        n = len(self.tr.frames)
        due = (n - self.last_bg >= 10) if n <= 80 else (n - self.last_bg >= 30)
        if due and n >= 8:
            t0 = now()
            self.tr.rebuild_background()
            self.tr.retrack()
            self.last_bg = n
            if not self.target_fixed or n <= 80:
                self.find_target()
            self.log("bg", n=n, dt=round(now() - t0, 3), size=round(self.tr.cube_size(), 2),
                     target=self.target_px, missing=int(sum(p is None for p in self.tr.pos)))

    def act(self, row):
        o = self.rb.step([row])
        self.ingest(o)
        return o

    def hold_row(self, target=None, gain=0.6):
        """Row that holds/approaches target (world) with P-control; zero rotation; gripper closed."""
        e = self.ee()
        if target is None or e is None:
            return [0, 0, 0, 0, 0, 0, GRIP]
        d = (np.asarray(target) - e) / 0.05 * gain
        n = np.linalg.norm(d)
        if n > TRAVEL_MAX:
            d = d * TRAVEL_MAX / n
        return [d[0], d[1], d[2], 0, 0, 0, GRIP]

    def time_left(self):
        return WALL_LIMIT - (now() - self.wall0)

    # ------------------------------------------------------------ target / arrivals
    def find_target(self):
        """Target = static non-table component of the background closest to the cube path."""
        tr = self.tr
        if tr is None or tr.bg is None:
            return
        bg = np.clip(tr.bg, 0, 255).astype(np.uint8)
        comps, lab = pc.nontable_components(bg, self.top_row)
        P = tr.positions_array()
        ok = ~np.isnan(P[:, 0])
        if ok.sum() < 8:
            return
        Pv = P[ok]
        cands = []
        for c in comps:
            if c["touches_top"] or c["area"] < 25 or c["area"] > 3000:
                continue
            if self.button_px is not None and math.hypot(c["cx"] - self.button_px[0], c["cy"] - self.button_px[1]) < 12:
                continue
            dmin = float(np.min(np.hypot(Pv[:, 0] - c["cx"], Pv[:, 1] - c["cy"])))
            cands.append((dmin, c))
        if not cands:
            return
        cands.sort(key=lambda x: x[0])
        dmin, c = cands[0]
        size = tr.cube_size()
        if dmin < 0.8 * size:
            new = (c["cx"], c["cy"])
            if self.target_px is None or math.hypot(new[0] - self.target_px[0], new[1] - self.target_px[1]) > 0.5:
                self.log("target", px=new, dmin=round(dmin, 2), area=c["area"], sat=round(c["frac_sat"], 2),
                         others=[(round(d, 1), round(k["cx"]), round(k["cy"])) for d, k in cands[1:4]])
            self.target_px = new
            if len(Pv) > 60:
                self.target_fixed = True

    def arrivals_state(self):
        """(done_count, events, P, D_now, size)."""
        tr = self.tr
        P = tr.positions_array()
        size = tr.cube_size()
        if self.target_px is None or len(P) == 0:
            return 0, [], P, None, size
        ev = pc.find_arrivals(P, self.target_px, size)
        t = len(P) - 1
        D = pc.distance_series(P, self.target_px)
        dnow = None if np.isnan(D[t]) else float(D[t])
        done = 0
        for e in ev:
            if e["closed"]:
                done += 1
            else:
                # open excursion: passed its minimum if distance has grown since
                if dnow is not None and e["i_min"] < t and dnow > e["d_min"] + 0.15 * size:
                    done += 1
        return done, ev, P, dnow, size

    def predict_next(self, P, ev, size):
        """Predict closest-approach time of the next (upcoming) arrival. Returns dict or None."""
        t = len(P) - 1
        valid = np.nonzero(~np.isnan(P[:, 0]))[0]
        if len(valid) == 0:
            return None
        t0 = int(valid[-1])
        if t - t0 > 3:
            return None
        prev = [e["t_min"] for e in ev if e["closed"] and e["t_min"] < t0 - 3]
        best = None
        if prev:
            try:
                r = pc.predict_replay(P, self.target_px, t0, prev[-6:], w=6)
            except Exception:
                r = None
            if r is not None and r["cost"] < 1.0 + 0.04 * size:
                best = r
        if best is None:
            q = pc.predict_quadratic(P, self.target_px, t0, w=4)
            if q is not None and q["d"] < 0.6 * size and q["t"] - t0 < 9:
                best = q
        return best

    # ------------------------------------------------------------ press timing
    def k_reg(self, z_now):
        """Fractional number of press rows (from rest) until registration."""
        zr = self.z_reg if self.z_reg is not None else (self.button[2] - REG_DEPTH)
        need = max(0.0, z_now - zr)
        prof = self.profile or [0.0, 0.012, 0.035, 0.065, 0.10, 0.14]
        for k in range(1, len(prof)):
            if prof[k] >= need:
                a, b = prof[k - 1], prof[k]
                return (k - 1) + (need - a) / max(1e-6, b - a)
        v = max(0.01, prof[-1] - prof[-2])
        return (len(prof) - 1) + (need - prof[-1]) / v

    def above_button(self, tol=0.015):
        e = self.ee()
        if e is None or self.button is None:
            return False
        return float(np.hypot(*(e[:2] - self.button[:2]))) < tol

    def check_trigger(self):
        """Return True when the press must start now. Logs the decision."""
        if self.target_px is None or self.tr is None or self.button is None:
            return False
        done, ev, P, dnow, size = self.arrivals_state()
        t = self.rb.t
        info = {"step": t, "done": done, "N": self.N, "d": None if dnow is None else round(dnow, 1)}
        if done > self.N - 1:
            # missed / over-counted: press if still inside the excursion of the N-th
            last = ev[-1] if ev else None
            inside = last is not None and not last["closed"]
            info["late"] = True
            info["inside"] = inside
            self.decision = info
            return inside and done == self.N
        if done < self.N - 1:
            self.decision = info
            return False
        pred = self.predict_next(P, ev, size)
        if pred is None:
            self.decision = info
            return False
        # the prediction must refer to a new excursion, not the one already counted
        done_ts = []
        for e_ in ev:
            if e_["closed"] or (dnow is not None and e_["i_min"] < t and dnow > e_["d_min"] + 0.15 * size):
                done_ts.append(e_["t_min"])
        if (done_ts and pred["t"] < done_ts[-1] + 5.0) or pred["t"] < t - 0.5:
            info["stale_pred"] = round(pred["t"], 2)
            self.decision = info
            return False
        e = self.ee()
        kr = self.k_reg(e[2]) if e is not None else 2.0
        t_reg_now = t + math.ceil(kr - 0.05)   # registration is detected at an integer step
        goal = pred["t"] - LEAD
        info.update({"pred": round(pred["t"], 2), "m": pred["method"], "cost": round(pred.get("cost", -1), 2),
                     "kr": round(kr, 2), "goal": round(goal, 2)})
        self.decision = info
        return t_reg_now >= goal - 0.5

    # ------------------------------------------------------------ phases
    def init_phase(self):
        o = self.rb.observe()
        st = self.rb.status()
        ins = o.instruction or (st.instruction if st else None)
        n = iu.parse_count(ins)
        self.N = n if n else DEFAULT_N
        if o.max_steps is None and st is not None:
            o.max_steps = st.max_steps
        self.log("init", version=VERSION, instruction=ins, N=self.N, parsed=n, obs=o.summary(),
                 status=None if st is None else st.summary())
        # front/wrist sanity: front has no dark gripper pixels at the bottom
        if o.front is not None and o.wrist is not None:
            sf, sw = pc.dark_bottom_score(o.front), pc.dark_bottom_score(o.wrist)
            if sf > sw + 0.01:
                self.rb.swap = True
                o.front, o.wrist = o.wrist, o.front
            self.log("cam_check", dark_front=round(sf, 4), dark_wrist=round(sw, 4), swap=self.rb.swap)
        if o.front is None:
            # try the initial frame-sequence manifest
            mp = iu.frame_sequence_manifest(o.meta)
            if mp:
                fr = iu.manifest_front_images(mp)
                if fr:
                    o.front = fr[-1][3]
                    o.front_src = "manifest"
            self.log("front_fallback", ok=o.front is not None, manifest=mp)
        if o.front is not None:
            save_png(o.front, "front_t0.png")
        if o.wrist is not None:
            save_png(o.wrist, "wrist_t0.png")
        f0 = o.front
        self.top_row = pc.table_top_row(f0) if f0 is not None else 12
        self.tr = pc.CubeTracker(self.top_row)
        self.tr.add(f0)
        self.tr.pos.append(None)
        self.tr.areas.append(None)
        # 4 zero steps (gripper closing) to separate moving from static components
        for _ in range(4):
            self.act([0, 0, 0, 0, 0, 0, GRIP])
            if self.rb.ended:
                return
        f4 = self.rb.obs.front if self.rb.obs is not None else None
        self.identify(f0, f4)

    def identify(self, f0, f4):
        if f0 is None or f4 is None:
            self.log("identify_fail", why="no front image")
            return
        c0, lab0 = pc.nontable_components(f0, self.top_row)
        c4, lab4 = pc.nontable_components(f4, self.top_row)
        stat, moving = [], []
        for c in c0:
            if c["touches_top"]:
                continue
            near = min(c4, key=lambda k: math.hypot(k["cx"] - c["cx"], k["cy"] - c["cy"])) if c4 else None
            disp = 99.0 if near is None else math.hypot(near["cx"] - c["cx"], near["cy"] - c["cy"])
            da = 1.0 if near is None else abs(near["area"] - c["area"]) / max(1, c["area"])
            c["disp"] = disp
            if disp < 0.8 and da < 0.15:
                stat.append(c)
            else:
                moving.append(c)
        # heights of static comps via the front depth
        pts, meta = [], []
        for c in stat:
            g = pc.gray_centroid(f0, c, lab0)
            cu, cv = g if g else (c["cx"], c["cy"])
            c["probe"] = (cu, cv)
            pts.append((cu, cv))
            meta.append(("obj", c))
            # table ring below/left/right of the component
            w = max(c["x1"] - c["x0"], 6)
            for du, dv in ((-w - 4, 0), (w + 4, 0), (0, (c["y1"] - c["y0"]) + 5)):
                pts.append((float(np.clip(c["cx"] + du, 1, 254)), float(np.clip(c["cy"] + dv, 1, 254))))
                meta.append(("table", c))
        # a few pure table probes for the table plane
        world = self.rb.p2p("front", pts) if pts else []
        # obs used for p2p is the latest (step 4); static comps unchanged so pixels still valid
        for c in stat:
            c["z"], tz = None, []
            for (kind, cc), wpt in zip(meta, world):
                if cc is not c or wpt is None:
                    continue
                if kind == "obj":
                    c["z"], c["world"] = float(wpt[2]), wpt
                else:
                    tz.append(float(wpt[2]))
            c["table_z"] = float(np.median(tz)) if tz else None
            c["h"] = (c["z"] - c["table_z"]) if (c["z"] is not None and c["table_z"] is not None) else None
            c["bscore"] = (100.0 * c["h"] if c["h"] is not None else 0.0) + 3 * c["frac_white"] + 3 * c["frac_gray"] - 4 * c["frac_sat"]
        self.static = stat
        self.log("identify", static=[{k: c.get(k) for k in ("area", "cx", "cy", "disp", "frac_white", "frac_gray",
                                                             "frac_sat", "z", "table_z", "h", "bscore", "world")}
                                     for c in stat],
                 moving=[{k: c.get(k) for k in ("area", "cx", "cy", "disp")} for c in moving])
        if not stat:
            return
        b = max(stat, key=lambda c: c["bscore"])
        self.button_px = b["probe"]
        if b.get("world") is not None:
            self.button = np.array(b["world"], float)
            # front depth hits the cap surface; keep cap z as measured
        self.tr.cube_xy0 = (moving[0]["cx"], moving[0]["cy"]) if moving else None
        self.tr.cube_area0 = moving[0]["area"] if moving else None
        # initial target guess: static, not the button, most saturated
        others = [c for c in stat if c is not b]
        if others:
            tg = max(others, key=lambda c: c["frac_sat"] - (c["h"] or 0) * 50)
            self.target_px = (tg["cx"], tg["cy"])
        self.log("button", world=self.button, px=self.button_px, target_guess=self.target_px)

    def approach_phase(self):
        if self.button is None:
            self.log("approach_skip", why="no button")
            return False
        tgt = self.button.copy()
        tgt[2] = self.button[2] + Z_ABOVE
        e = self.ee()
        start_z = e[2] if e is not None else tgt[2]
        for i in range(80):
            if self.rb.ended:
                return False
            e = self.ee()
            if e is None:
                self.act([0, 0, 0, 0, 0, 0, GRIP])
                continue
            # stay high until xy is close
            dxy = float(np.hypot(*(tgt[:2] - e[:2])))
            sub = tgt.copy()
            if dxy > 0.03:
                sub[2] = max(tgt[2], min(start_z, tgt[2] + 0.1))
            row = self.hold_row(sub, gain=0.7)
            self.act(row)
            self.log("step", ph="APPROACH", t=self.rb.t, ee=self.ee(), dxy=round(dxy, 4), row=[round(x, 3) for x in row[:3]])
            e = self.ee()
            if e is not None and np.linalg.norm(tgt - e) < 0.008:
                break
            if self.urgent():
                return True
        return True

    def settle(self, target, n_max=8, tol=0.0015):
        prev = None
        for _ in range(n_max):
            if self.rb.ended:
                return
            self.act(self.hold_row(target, gain=0.6))
            e = self.ee()
            if prev is not None and e is not None and np.linalg.norm(e - prev) < tol and np.linalg.norm(e - target) < 0.006:
                return
            prev = e
            if self.urgent():
                return

    def wrist_locate(self):
        """Use the wrist camera to measure the cap position. Returns world point or None."""
        o = self.rb.obs
        if o is None or o.wrist is None:
            return None
        wb = pc.wrist_button(o.wrist)
        if wb is None:
            self.log("wrist_none")
            return None
        e = self.ee()
        if e is not None:
            self.widths.append((float(e[2]), wb["width"]))
        pts = []
        if wb["cap"] is not None:
            pts.append((wb["cap"][0], wb["cap"][1]))
        pts.append((wb["sq_cx"], wb["sq_cy"]))
        x0, y0, x1, y1 = wb["sq_bbox"]
        pts += [(x0 + 2, y0 + 2), (x1 - 2, y0 + 2)]
        w = self.rb.p2p("wrist", pts)
        self.log("wrist", wb={k: wb[k] for k in ("sq_cx", "sq_cy", "width", "cap", "sq_area")}, world=w, ee=e)
        if not w or w[0] is None:
            return None
        cap = w[0] if wb["cap"] is not None else None
        base = [p for p in w[1 if wb["cap"] is not None else 0:] if p is not None]
        if cap is None and base:
            cap = base[0]
        if cap is None:
            return None
        res = np.array(cap, float)
        if wb["cap"] is None:
            res[2] += 0.01   # only base seen; cap top is higher
        return res

    def refine_phase(self):
        p = self.wrist_locate()
        if p is None or self.button is None:
            return
        d = p - self.button
        ok = float(np.hypot(d[0], d[1])) < 0.06 and abs(d[2]) < 0.04
        self.log("refine", wrist=p, front=self.button, d=d, ok=ok)
        if ok:
            self.button = np.array(p, float)
            tgt = self.button + np.array([0, 0, Z_ABOVE])
            self.settle(tgt, n_max=10)

    def calib_phase(self):
        """Free-air descent profile from rest with dz=-1 rows."""
        e0 = self.ee()
        if e0 is None:
            return
        prof = [0.0]
        z0 = e0[2]
        floor = self.button[2] + 0.05
        for k in range(1, 7):
            if self.rb.ended:
                return
            self.act([0, 0, -1.0, 0, 0, 0, GRIP])
            e = self.ee()
            prof.append(float(z0 - e[2]))
            if e[2] <= floor + 0.02 or prof[-1] >= 0.06:
                break
        # stop and settle
        post = []
        for _ in range(4):
            self.act([0, 0, 0, 0, 0, 0, GRIP])
            e = self.ee()
            post.append(float(z0 - e[2]))
        if len(prof) >= 3 and all(prof[i + 1] >= prof[i] - 1e-4 for i in range(len(prof) - 1)) and prof[-1] > 0.01:
            self.profile = prof
        self.log("calib", profile=prof, post=post, used=self.profile is not None)
        # width at this lower height for scale estimate
        self.wrist_locate()

    def estimate_zreg(self):
        prior = self.button[2] - REG_DEPTH
        est = None
        if len(self.widths) >= 2:
            (za, wa), (zb, wb) = self.widths[0], self.widths[-1]
            if abs(za - zb) > 0.02 and abs(wa - wb) > 3:
                K = (wb * zb - wa * za) / (wa - wb)
                kk = wa * (za + K)
                est = kk / EXPERT_W_REG - K
        use = prior
        if est is not None and abs(est - prior) < 0.015:
            use = 0.5 * (est + prior)
        self.z_reg = use
        self.log("zreg", prior=prior, expert=est, used=use, widths=self.widths)

    def hover_phase(self):
        tgt = self.button + np.array([0, 0, H_HOVER])
        self.settle(tgt, n_max=14, tol=0.001)
        p = self.wrist_locate()
        if p is not None:
            d = p - self.button
            if float(np.hypot(d[0], d[1])) < 0.02 and abs(d[2]) < 0.015:
                # small xy correction only (z from the higher view is more complete)
                self.button[:2] = 0.5 * (self.button[:2] + p[:2])
                tgt = self.button + np.array([0, 0, H_HOVER])
                self.settle(tgt, n_max=6, tol=0.001)

    def urgent(self):
        """True if the N-th arrival is imminent and we are above the button: trigger press logic."""
        if self.button is None or not self.above_button(0.02):
            return False
        try:
            return self.check_trigger()
        except Exception as e:
            self.log("trigger_err", err=str(e)[:300])
            return False

    def est_time_to_nth(self):
        done, ev, P, dnow, size = self.arrivals_state()
        ts = [e["t_min"] for e in ev if e["closed"]]
        if len(ts) >= 2:
            inter = float(np.median(np.diff(ts)))
            return ts[-1] + (self.N - done) * inter - self.rb.t, done
        return None, done

    def wait_phase(self):
        hold = self.button + np.array([0, 0, H_HOVER])
        max_steps = 2000
        for i in range(max_steps):
            if self.rb.ended:
                return False
            try:
                trig = self.check_trigger()
            except Exception as ex:
                self.log("trigger_err", err=str(ex)[:300], tb=traceback.format_exc()[-800:])
                trig = False
            if trig:
                self.log("trigger", **self.decision)
                return True
            if i % 5 == 0 or self.decision.get("pred") is not None:
                self.log("wait", ee=self.ee(), **self.decision)
            self.act(self.hold_row(hold, gain=0.5))
            if self.time_left() < 30:
                self.log("wall_guard")
                return False
        return False

    def press_phase(self):
        e = self.ee()
        self.pressed_at = self.rb.t
        self.log("press_start", t=self.rb.t, ee=e, decision=self.decision, z_reg=self.z_reg, profile=self.profile)
        zs = []
        for k in range(PRESS_MAX):
            if self.rb.ended:
                break
            o = self.rb.step([[0, 0, -1.0, 0, 0, 0, GRIP]])
            self.ingest(o)
            e2 = self.ee()
            zs.append(None if e2 is None else round(float(e2[2]), 4))
            self.log("press", k=k + 1, t=self.rb.t, ee=e2, ended=self.rb.ended, complete=self.rb.complete)
        # hold still (do not release immediately)
        for _ in range(6):
            if self.rb.ended:
                break
            o = self.rb.step([[0, 0, 0, 0, 0, 0, GRIP]])
            self.ingest(o)
        self.log("press_end", t=self.rb.t, zs=zs, ended=self.rb.ended, complete=self.rb.complete)

    # ------------------------------------------------------------ main
    def run(self):
        self.init_phase()
        if self.rb.ended:
            return
        self.phase = "APPROACH"
        self.approach_phase()
        if self.button is None:
            # nothing sensible to do; hold until end of wall budget is not useful -> idle a little
            self.log("abort", why="button not localised")
            for _ in range(20):
                if self.rb.ended:
                    return
                self.act([0, 0, 0, 0, 0, 0, GRIP])
            return
        if not self.urgent():
            self.phase = "REFINE"
            eta, done = self.est_time_to_nth()
            self.log("eta", eta=eta, done=done)
            if eta is None or eta > 20:
                self.refine_phase()
            eta, done = self.est_time_to_nth()
            if (eta is None or eta > 32) and not self.urgent():
                self.phase = "CALIB"
                self.calib_phase()
            self.estimate_zreg()
            if not self.urgent():
                self.phase = "HOVER"
                self.hover_phase()
        else:
            self.estimate_zreg()
        self.phase = "WAIT"
        go = self.wait_phase()
        if go:
            self.phase = "PRESS"
            self.press_phase()
        self.log("done", t=self.rb.t, ended=self.rb.ended, complete=self.rb.complete,
                 mean_call=round(self.rb.mean_call(), 3), wall=round(now() - self.wall0, 1))


def main(ctx):
    c = None
    try:
        c = Controller(ctx)
        c.run()
    except Exception as e:  # never crash
        try:
            msg = traceback.format_exc()
            print(json.dumps({"kind": "fatal", "err": str(e)[:500], "tb": msg[-3000:]}), flush=True)
        except Exception:
            pass
        # safe fallback: if we know enough, still try a timed press loop
        try:
            if c is not None and not c.rb.ended and c.button is not None and c.tr is not None:
                c.log("fallback_wait")
                if c.wait_phase():
                    c.press_phase()
        except Exception:
            pass
    return None
