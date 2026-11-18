"""Robot I/O: defensive packet parsing, stepping, logging.

All packet layouts are discovered defensively (recursive key search), because the
controller had to be written before any live packet was seen (R1 budget).
"""
import json
import os
import time
import traceback

import numpy as np

import geom

OUT_DIR = "/workspace/output"
WS = "/workspace"
POS_SCALE = 0.05   # m per unit action (docs)
ROT_SCALE = 0.5    # rad per unit action (docs)

_T0 = time.time()
_LOGF = None


def elapsed():
    return time.time() - _T0


def log(*parts):
    global _LOGF
    msg = "[%7.1fs] " % elapsed() + " ".join(str(p) for p in parts)
    try:
        print(msg, flush=True)
    except Exception:
        pass
    try:
        if _LOGF is None:
            os.makedirs(OUT_DIR, exist_ok=True)
            _LOGF = open(os.path.join(OUT_DIR, "controller_log.txt"), "a")
        _LOGF.write(msg + "\n")
        _LOGF.flush()
    except Exception:
        pass


def save_json(name, obj, limit=400000):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        s = json.dumps(obj, default=str)
        if len(s) > limit:
            s = s[:limit]
        with open(os.path.join(OUT_DIR, name), "w") as f:
            f.write(s)
    except Exception:
        pass


# ---------------------------------------------------------------- parsing

def text_json(reply):
    """First JSON text block of an MCP result."""
    if isinstance(reply, dict):
        for b in reply.get("content", []) or []:
            if isinstance(b, dict) and b.get("type") == "text":
                try:
                    return json.loads(b.get("text", ""))
                except Exception:
                    continue
        if "structuredContent" in reply and isinstance(reply["structuredContent"], dict):
            return reply["structuredContent"]
    return {}


def find_all(obj, key, path=(), out=None, depth=0):
    if out is None:
        out = []
    if depth > 12:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append((path + (k,), v))
            find_all(v, key, path + (k,), out, depth + 1)
    elif isinstance(obj, list) and len(obj) < 5000:
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                find_all(v, key, path + (i,), out, depth + 1)
    return out


def _is_vec(v, n):
    try:
        return isinstance(v, (list, tuple)) and len(v) == n and all(
            isinstance(x, (int, float)) and np.isfinite(x) for x in v)
    except Exception:
        return False


def find_state(pkt):
    """The current 8-D state: prefer the shortest path that is not inside a list of steps."""
    cands = [(p, v) for p, v in find_all(pkt, "state") if _is_vec(v, 8)]
    if not cands:
        return None
    def score(item):
        p = item[0]
        in_list = any(isinstance(x, int) for x in p)
        obs = any(isinstance(x, str) and "observ" in x for x in p)
        return (in_list, 0 if obs else 1, len(p))
    cands.sort(key=score)
    p, v = cands[0]
    if any(isinstance(x, int) for x in p):
        # inside a list (e.g. per-step records): take the last element of that list
        best = None
        for pp, vv in find_all(pkt, "state"):
            if _is_vec(vv, 8):
                best = vv
        v = best
    return np.array(v, dtype=float)


def find_bool(pkt, key):
    vals = [(len(p), v) for p, v in find_all(pkt, key) if isinstance(v, bool)]
    if not vals:
        return None
    vals.sort(key=lambda x: x[0])
    # any True at any depth counts for terminal flags
    if any(v for _, v in vals):
        return True
    return vals[0][1]


def find_step_id(pkt):
    vals = [(len(p), v) for p, v in find_all(pkt, "step_id") if isinstance(v, int)]
    if not vals:
        return None
    vals.sort(key=lambda x: x[0])
    return vals[0][1]


def find_limits(pkt):
    """Return (remaining_steps or None, raw limits dict)."""
    lims = find_all(pkt, "episode_limits")
    raw = lims[0][1] if lims else {}
    rem = None
    def walk(o):
        nonlocal rem
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and "remain" in kl and (
                        "step" in kl or kl == "remaining") and "call" not in kl:
                    rem = int(v) if rem is None else min(rem, int(v))
                walk(v)
    walk(raw if raw else pkt)
    if rem is None and isinstance(raw, dict):
        mx = None
        cur = None
        for k, v in raw.items():
            kl = str(k).lower()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if ("max" in kl or "limit" in kl) and "step" in kl and "call" not in kl and "row" not in kl:
                    mx = int(v)
                if kl in ("step", "steps", "step_id", "elapsed_steps", "steps_used", "used_steps"):
                    cur = int(v)
        if mx is not None:
            rem = mx - (cur or 0)
    return rem, raw


# ---------------------------------------------------------------- robot wrapper

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.complete = False
        self.remaining = None
        self.steps_sent = 0
        self.step_calls = 0
        self.step_time = 0.0
        self.last_pkt = None
        self.rot_mode = "world"  # composition convention for rotation deltas
        self._first_step_saved = False
        self.deadline = 1050.0     # controller seconds after which we stop moving
        self.step_errors = 0
        self.consecutive_errors = 0
        self.halted = False

    # -- observation
    def observe(self):
        r = self.ctx.call("robomme_observe")
        pkt = text_json(r)
        self.last_pkt = pkt
        self._absorb(pkt)
        return pkt, r

    def status(self):
        try:
            r = self.ctx.call("robomme_status")
            pkt = text_json(r)
            self._absorb(pkt, state_ok=False)
            return pkt
        except Exception as e:
            log("status failed", repr(e))
            return {}

    def _absorb(self, pkt, state_ok=True):
        if state_ok:
            s = find_state(pkt)
            if s is not None:
                self.state = s
        e = find_bool(pkt, "episode_ended")
        if e:
            self.ended = True
        c = find_bool(pkt, "task_complete")
        if c:
            self.complete = True
        rem, _ = find_limits(pkt)
        if rem is not None:
            self.remaining = rem

    def pos(self):
        return self.state[:3].copy()

    def R(self):
        return geom.rotvec_to_mat(self.state[3:6])

    def width(self):
        return float(self.state[6] + self.state[7])

    def can_move(self):
        if self.ended or self.complete or self.halted:
            return False
        if elapsed() > self.deadline:
            return False
        # `remaining` is parsed heuristically; never stop on it alone (the env ends the episode)
        return True

    # -- stepping
    def step_rows(self, rows):
        if not self.can_move():
            return False
        rows = [[float(np.clip(x, -1, 1)) for x in row] for row in rows]
        if self.remaining is not None and self.remaining > 0:
            rows = rows[:self.remaining]
        if not rows:
            return False
        t = time.time()
        try:
            r = self.ctx.call("robomme_step", actions=rows)
        except Exception as e:
            log("STEP ERROR", repr(e)[:500])
            self.step_errors += 1
            self.consecutive_errors += 1
            if self.consecutive_errors >= 6:
                log("too many consecutive step errors -> stop moving")
                self.halted = True
            self.step_time += time.time() - t
            # ambiguous: re-observe before continuing
            try:
                self.observe()
            except Exception as e2:
                log("observe after step error failed", repr(e2)[:300])
            self.status()
            return False
        self.step_time += time.time() - t
        self.step_calls += 1
        self.consecutive_errors = 0
        pkt = text_json(r)
        self.last_pkt = pkt
        if not self._first_step_saved:
            save_json("first_step_reply.json", pkt)
            self._first_step_saved = True
        before = self.remaining
        s = find_state(pkt)
        self._absorb(pkt)
        if s is None:
            try:
                self.observe()
            except Exception as e:
                log("observe failed", repr(e)[:300])
        self.steps_sent += len(rows)
        if self.remaining is not None and before is not None and self.remaining == before:
            self.remaining = before - len(rows)
        return True

    def action_to(self, p_tgt, R_tgt, grip, max_pos=1.0, max_rot=1.0):
        """One action row driving the measured pose to (p_tgt, R_tgt)."""
        p = self.pos()
        dp = (np.asarray(p_tgt) - p) / POS_SCALE
        n = np.linalg.norm(dp)
        if n > max_pos:
            dp = dp * (max_pos / n)
        Rm = self.R()
        if R_tgt is None:
            dr = np.zeros(3)
        else:
            if self.rot_mode == "world":
                dr = geom.mat_to_rotvec(R_tgt @ Rm.T) / ROT_SCALE
            else:
                dr = geom.mat_to_rotvec(Rm.T @ R_tgt) / ROT_SCALE
            nr = np.linalg.norm(dr)
            if nr > max_rot:
                dr = dr * (max_rot / nr)
        return [dp[0], dp[1], dp[2], dr[0], dr[1], dr[2], float(grip)]

    def goto(self, p_tgt, R_tgt, grip, tol=0.004, rtol=0.03, max_steps=60, max_pos=0.4,
             max_rot=0.4, label=""):
        """Closed-loop move; returns True when within tolerance."""
        last_err = None
        last_rerr = None
        stall = 0
        for i in range(max_steps):
            if not self.can_move():
                return False
            e = float(np.linalg.norm(np.asarray(p_tgt) - self.pos()))
            er = 0.0 if R_tgt is None else geom.rot_angle(R_tgt, self.R())
            if e < tol and er < rtol:
                return True
            if last_err is not None:
                pos_stuck = e > last_err - 0.0005 or e < tol
                rot_stuck = er > last_rerr - 0.003 or er < rtol
                stall = stall + 1 if (pos_stuck and rot_stuck) else 0
                if stall >= 6:
                    log("goto stall", label, "err=%.4f rerr=%.3f" % (e, er))
                    return False
            last_err, last_rerr = e, er
            row = self.action_to(p_tgt, R_tgt, grip, max_pos=max_pos, max_rot=max_rot)
            self.step_rows([row])
            if self.state is None:
                return False
            er2 = 0.0 if R_tgt is None else geom.rot_angle(R_tgt, self.R())
            self._check_rot_convention(er, er2)
        e = float(np.linalg.norm(np.asarray(p_tgt) - self.pos()))
        log("goto timeout", label, "err=%.4f" % e)
        return e < 2 * tol

    _rot_bad = 0
    _rot_switched = False

    def _check_rot_convention(self, before, after):
        """If rotation error repeatedly grows under rotation commands, flip composition."""
        if before < 0.04 or before > 2.6:
            return
        if after > before + 0.01:
            self._rot_bad += 1
        elif after < before - 0.01:
            self._rot_bad = max(0, self._rot_bad - 1)
        if self._rot_bad >= 3 and not self._rot_switched:
            self.rot_mode = "body" if self.rot_mode == "world" else "world"
            self._rot_switched = True
            self._rot_bad = 0
            log("ROTATION CONVENTION SWITCHED to", self.rot_mode)

    def probe_rotation(self, grip=-1.0):
        """Command small world-z yaws and see whether the measured rotation follows the world
        or the body interpretation of the action's rotation part. Returns a diagnostics dict."""
        out = {}
        try:
            if self.state is None or not self.can_move():
                return out
            R0 = self.R()
            p0 = self.pos()
            amt = 0.2  # 0.1 rad requested
            self.step_rows([[0, 0, 0, 0, 0, amt, grip]])
            self.step_rows([[0, 0, 0, 0, 0, 0, grip]])
            R1 = self.R()
            w = mat_to_rotvec_safe(R1 @ R0.T)
            hyp_w = np.array([0, 0, 1.0])
            hyp_b = R0 @ np.array([0, 0, 1.0])
            sw = float(np.dot(w, hyp_w))
            sb = float(np.dot(w, hyp_b))
            out = {"measured_world_delta": np.round(w, 4).tolist(), "proj_world": sw, "proj_body": sb}
            if np.linalg.norm(w) > 0.02:
                rw = np.linalg.norm(w - hyp_w * sw)
                rb = np.linalg.norm(w - hyp_b * sb)
                body = (sw <= 0 < sb) or (sw > 0 and sb > 0 and rb < rw - 0.01)
                if body:
                    self.rot_mode = "body"
                    self._rot_switched = True
                out["mode"] = self.rot_mode
                out["gain"] = (sw if self.rot_mode == "world" else sb) / (amt * ROT_SCALE)
            # undo the yaw (closed loop, current convention)
            self.goto(p0, R0, grip, tol=0.003, rtol=0.01, max_steps=10, max_pos=0.2, max_rot=0.3,
                      label="probe-undo")
            out["undo_rot_err"] = geom.rot_angle(R0, self.R())
        except Exception as e:
            out["error"] = repr(e)
        log("rotation probe:", out)
        return out

    def hold(self, grip, n):
        """Hold the pose while commanding the gripper; return finger widths."""
        widths = []
        for _ in range(n):
            if not self.can_move():
                break
            self.step_rows([[0, 0, 0, 0, 0, 0, float(grip)]])
            if self.state is not None:
                widths.append(self.width())
        return widths

    def grip_settle(self, p, R, grip, min_n=6, max_n=25):
        widths = []
        for i in range(max_n):
            if not self.can_move():
                break
            row = self.action_to(p, R, grip, max_pos=0.2, max_rot=0.2)
            self.step_rows([row])
            if self.state is None:
                break
            widths.append(self.width())
            if i + 1 >= min_n and len(widths) >= 3 and max(widths[-3:]) - min(widths[-3:]) < 0.0004:
                break
        return widths


def mat_to_rotvec_safe(R):
    return geom.mat_to_rotvec(R)


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        log("EXCEPTION in", getattr(fn, "__name__", fn), traceback.format_exc()[-1500:])
        return None
