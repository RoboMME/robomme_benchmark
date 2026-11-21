"""PatternLock controller (RAX, interaction-only, written blind: no live packet was seen before the rehearsal).

Pipeline (each phase guarded, main never raises):
  P0 observe      -> packet, state, reset frame_sequence manifest
  P1 plan         -> demo stick trajectory from the conditioning video
                     A_state : per-frame `state` EE positions in the manifest (primary)
                     A_vision: lowest foreground point in front depth (+TCP offset from exec frame)
                     B_touch : static robot -> changed (highlighted) targets ordered by first change
                     none    -> hold and exit cleanly
  P2 execute      -> closed-loop waypoint follower (world-frame deltas, chunked rows,
                     keypoints must be reached by measurement, stall -> accept and continue)
  P3 settle       -> hold a few steps watching for task_complete, then return.
Transitions: P0 fail -> exit; P1 none -> P3; P2 episode_ended/task_complete -> exit;
P2 wall/step budget exhausted -> exit; P2 done -> P3 -> exit.
"""
import json
import math
import time
import traceback

from plcore import (log, save_json, elapsed, reply_json, find_all, get_state, get_flag, get_first,
                    load_manifest, split_frames, frame_time, is_num_list, vadd, vsub, vscale, vnorm,
                    clip_norm, rot_error, rot_angle_between, build_waypoints, path_length)

VERSION = "patternlock-v1"

# ---- tunables (untested live: chosen from the documented action scale 1.0 = 0.05 m, 0.5 rad)
POS_SCALE = 0.05
ROT_SCALE = 0.5
V_DRAW = 0.008        # m/step near drawing height (within 3 cm of the lowest demo z)
V_TRANSIT = 0.015     # m/step elsewhere
V_MAX_CAP = 0.025
TOL_WP = 0.006        # m, pass-through waypoints (judged on predicted position)
TOL_KEY = 0.004       # m, keypoints (judged on measured position)
KEY_MAX_STEPS = 40    # steps spent on one keypoint before accepting it
ROT_MAX = 0.12        # rad per correction row
ROT_DEADBAND = 0.01   # rad, when tracking a varying demo orientation
ROT_DEADBAND_HOLD = 0.03  # rad, when holding the initial orientation
CHUNK = 5             # rows per call (adapted for wall time)
WALL_BUDGET_S = 900   # controller-side wall budget (limit 1200 s; episode 1800 s incl. startup)
SETTLE_STEPS = 10


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = None
        self.ended = False
        self.complete = False
        self.limits = None
        self.n_steps = 0
        self.call_times = []
        self.errors = 0
        self.trust_step_state = None   # decided after comparing step reply with observe
        self.last_packet = None

    def _flags(self, pk):
        if get_flag(pk, "episode_ended"):
            self.ended = True
        if get_flag(pk, "task_complete"):
            self.complete = True
        lim = get_first(pk, "episode_limits")
        if lim is not None:
            self.limits = lim

    def observe(self):
        t = time.time()
        res = self.ctx.call("robomme_observe")
        self.call_times.append(time.time() - t)
        pk = reply_json(res)
        self.last_packet = pk
        st = get_state(pk)
        if st is not None:
            self.state = st
        sid = get_first(pk, "step_id")
        if isinstance(sid, int):
            self.step_id = sid
        self._flags(pk)
        return pk

    def remaining_steps(self):
        lim = self.limits
        if isinstance(lim, dict):
            for k in ("remaining_steps", "steps_remaining", "remaining"):
                if isinstance(lim.get(k), (int, float)):
                    return int(lim[k])
            mx = None
            for k in ("max_steps", "step_limit", "max_episode_steps", "limit", "horizon"):
                if isinstance(lim.get(k), (int, float)):
                    mx = int(lim[k])
                    break
            used = None
            for k in ("steps", "step", "elapsed_steps", "used_steps", "step_id"):
                if isinstance(lim.get(k), (int, float)):
                    used = int(lim[k])
                    break
            if mx is not None:
                r = mx - (self.step_id if isinstance(self.step_id, int) else (used or 0))
                return r if r > 0 else None   # never trust a non-positive derived count
        return None

    def step(self, rows):
        if self.ended or not rows:
            return None
        rows = [[max(-1.0, min(1.0, float(x))) for x in r] for r in rows]
        t = time.time()
        try:
            res = self.ctx.call("robomme_step", actions=rows)
        except Exception as exc:
            self.errors += 1
            log("STEP ERROR", repr(exc)[:500])
            # physics may have advanced: re-read the robot instead of retrying
            try:
                st = self.ctx.call("robomme_status")
                self._flags(reply_json(st))
                log("status after error:", str(reply_json(st))[:400])
            except Exception as exc2:
                log("status failed", repr(exc2)[:300])
            try:
                self.observe()
            except Exception as exc3:
                log("observe failed", repr(exc3)[:300])
                self.errors += 1
            if self.errors >= 4:
                log("too many errors -> ending")
                self.ended = True
            return None
        self.call_times.append(time.time() - t)
        pk = reply_json(res)
        self.n_steps += len(rows)
        self._flags(pk)
        sid = get_first(pk, "step_id")
        st_reply = get_state(pk)
        if self.ended:
            if st_reply is not None:
                self.state = st_reply
            return pk
        if self.trust_step_state is True and st_reply is not None:
            self.state = st_reply
            if isinstance(sid, int):
                self.step_id = sid
            return pk
        try:
            self.observe()
        except Exception as exc:
            log("observe after step failed", repr(exc)[:300])
            if st_reply is not None:
                self.state = st_reply
            return pk
        if self.trust_step_state is None and st_reply is not None and self.state is not None:
            if max(abs(a - b) for a, b in zip(st_reply, self.state)) < 1e-6:
                self._agree = getattr(self, "_agree", 0) + 1
                if self._agree >= 3:
                    self.trust_step_state = True
                    log("step-reply state agrees with observe; skipping extra observes")
            else:
                self.trust_step_state = False
                log("step-reply state differs from observe; keep observing", st_reply[:3], self.state[:3])
        return pk


def gripper_cmd(state):
    if state is not None and len(state) >= 8 and (state[6] + state[7]) > 0.06:
        return -1.0
    return 1.0


# ---------------------------------------------------------------- planning
def plan_from_states(video):
    pts, rots, times = [], [], []
    for i, fr in enumerate(video):
        st = fr.get("state")
        if is_num_list(st) and len(st) >= 6:
            p = [float(x) for x in st[:3]]
            if max(abs(x) for x in p) > 5:
                continue
            pts.append(p)
            rots.append([float(x) for x in st[3:6]])
            times.append(frame_time(fr, i))
    info = {"n_states": len(pts)}
    if len(pts) < 3:
        return None, info
    rng = max(vnorm(vsub(p, pts[0])) for p in pts)
    info["range"] = rng
    zs = [p[2] for p in pts]
    info["z_min"], info["z_max"] = min(zs), max(zs)
    if rng < 0.02:
        return None, info
    rot_var = max(rot_angle_between(r, rots[0]) for r in rots)
    info["rot_var"] = rot_var
    return {"points": pts, "rots": rots, "times": times, "rot_var": rot_var}, info


def make_plan(ctx, robot, packet):
    manifest, mpath = load_manifest(packet)
    if manifest is None:
        log("no manifest found")
        return None
    video, execf = split_frames(manifest)
    exec_frame = execf[-1] if execf else None
    summ = {"manifest": str(mpath), "n_video": len(video), "n_exec": len(execf),
            "top_keys": list(manifest.keys())[:20],
            "frame_keys": list(video[0].keys()) if video else (list(execf[0].keys()) if execf else []),
            "first_video": {k: v for k, v in (video[0].items() if video else []) if k != "files"},
            "last_video": {k: v for k, v in (video[-1].items() if video else []) if k != "files"},
            "exec": {k: v for k, v in (exec_frame.items() if exec_frame else []) if k != "files"},
            "video_sampling": get_first(packet, "video_sampling")}
    log("manifest summary:", str(summ)[:1500])
    save_json("manifest_summary.json", summ)
    save_json("video_states.json", [[f.get("frame_index"), f.get("timestamp_s"), f.get("state")] for f in video])
    if not video:
        return None
    # ---- A_state
    state_plan = None
    try:
        plan, info = plan_from_states(video)
        log("A_state info:", info)
        if plan is not None:
            plan["mode"] = "A_state"
            suspicious = states_suspicious(video, exec_frame, robot.state)
            if not suspicious:
                return plan
            log("A_state plan suspicious; trying vision first, keeping states as fallback")
            state_plan = plan
    except Exception:
        log("A_state failed", traceback.format_exc()[-800:])
    vplan = vision_plan(ctx, robot, packet, video, exec_frame)
    if vplan is not None:
        return vplan
    return state_plan


def states_suspicious(video, exec_frame, exec_state):
    """Front depth of the first video frame identical to the execution frame (robot at the same
    pose) while the recorded states disagree by >3 cm -> the per-frame states are not trustworthy."""
    try:
        import plvision as V
        st0 = video[0].get("state")
        if exec_frame is None or exec_state is None or not is_num_list(st0):
            return False
        d0, de = V.frame_depth(video[0]), V.frame_depth(exec_frame)
        if d0 is None or de is None or d0.shape != de.shape:
            return False
        frac = float(((abs(d0 - de) > 0.01) & (d0 > 0) & (de > 0)).mean())
        dist = vnorm(vsub(st0[:3], exec_state[:3]))
        log("state sanity: depth-changed fraction %.4f, |state0-exec| %.4f m" % (frac, dist))
        return frac < 0.002 and dist > 0.03
    except Exception:
        log("state sanity check failed", traceback.format_exc()[-400:])
        return False


def vision_plan(ctx, robot, packet, video, exec_frame):
    # ---- vision fallbacks
    try:
        import plvision as V
    except Exception:
        log("vision import failed", traceback.format_exc()[-500:])
        return None
    model = None
    try:
        de = V.frame_depth(exec_frame) if exec_frame is not None else None
        if de is None:
            de = V.frame_depth(video[-1])
        if de is not None:
            model = V.calibrate_front(ctx, packet, de, robot.step_id if isinstance(robot.step_id, int) else 0)
        log("camera model:", model.describe() if model else None)
    except Exception:
        log("calibration failed", traceback.format_exc()[-800:])
    if model is None:
        return None
    try:
        traj, info = V.track_tip(video, exec_frame, model, robot.state)
        log("A_vision info:", str(info)[:800])
        if traj and info.get("range", 0) >= 0.03:
            pts = [p for _, p in traj]
            return {"mode": "A_vision", "points": pts, "rots": None, "times": [t for t, _ in traj],
                    "rot_var": 0.0, "table_z": info.get("table_z")}
    except Exception:
        log("A_vision failed", traceback.format_exc()[-800:])
    try:
        pts, info = V.detect_highlights(video, exec_frame, model)
        log("B info:", str(info)[:1500])
        if pts:
            de = V.frame_depth(exec_frame) if exec_frame is not None else None
            tz = V.table_height(model, de) if de is not None else None
            return {"mode": "B_touch", "targets": pts, "table_z": tz}
    except Exception:
        log("B failed", traceback.format_exc()[-800:])
    return None


# ---------------------------------------------------------------- execution
class Follower:
    def __init__(self, robot, rot_hold):
        self.robot = robot
        self.rot_hold = rot_hold          # fallback orientation target
        self.rot_enabled = True
        self.rot_err_hist = []
        self.init_rot_err = None
        self.chunk = CHUNK

    def rot_row(self, rot_target):
        st = self.robot.state
        if not self.rot_enabled or st is None or len(st) < 6:
            return [0.0, 0.0, 0.0]
        hold = rot_target is None
        tgt = rot_target if not hold else self.rot_hold
        if tgt is None:
            return [0.0, 0.0, 0.0]
        err = rot_error(tgt, st[3:6])
        ang = vnorm(err)
        if self.init_rot_err is None:
            self.init_rot_err = ang
        self.rot_err_hist.append(ang)
        h = self.rot_err_hist
        # divergence guards (a wrong rotation-frame assumption would make corrections grow the error)
        diverging = False
        if hold and ang > max(0.12, self.init_rot_err + 0.08):
            diverging = True
        if len(h) >= 6 and all(h[k + 1] >= h[k] for k in range(-6, -1)) and h[-1] - h[-6] > 0.06:
            diverging = True
        if ang > 0.4:
            diverging = True
        if diverging:
            log("orientation error diverging (%s): disabling rotation correction" % [round(x, 3) for x in h[-6:]])
            self.rot_enabled = False
            return [0.0, 0.0, 0.0]
        if ang < (ROT_DEADBAND_HOLD if hold else ROT_DEADBAND):
            return [0.0, 0.0, 0.0]
        corr = clip_norm(vscale(err, 0.8), ROT_MAX)
        return vscale(corr, 1.0 / ROT_SCALE)

    def adapt_chunk(self, remaining_wps):
        ct = self.robot.call_times[-10:]
        if not ct:
            return
        per_call = sum(ct) / len(ct) * (1 if self.robot.trust_step_state else 2)
        left = WALL_BUDGET_S - elapsed()
        need_calls = max(1, remaining_wps)
        if per_call * need_calls / max(self.chunk, 1) > 0.7 * left:
            self.chunk = min(12, self.chunk + 2)

    def follow(self, wps, v_of):
        """wps: list of dict(p, r, key). v_of(p_target) -> speed. Returns summary dict."""
        R = self.robot
        i = 0
        n = len(wps)
        key_steps = 0
        key_hist = []
        g = gripper_cmd(R.state)
        log("follow: %d waypoints (%d key), path %.3f m" % (n, sum(1 for w in wps if w["key"]), path_length(wps)))
        reached_keys, accepted_keys = 0, 0
        wd_i, wd_best, wd_chunks, skips = -1, 1e9, 0, 0   # stall watchdog for pass-through targets
        while i < n and not R.ended:
            if elapsed() > WALL_BUDGET_S:
                log("wall budget exhausted at wp", i)
                break
            rem = R.remaining_steps()   # only used to truncate rows; the episode end is authoritative
            p = R.state[:3]
            w = wps[i]
            if w["key"]:
                d = vnorm(vsub(w["p"], p))
                stalled = (len(key_hist) >= 3 and key_hist[-3] - d < 0.001 and d < 0.02)
                if d < TOL_KEY or stalled or key_steps >= KEY_MAX_STEPS:
                    if d < TOL_KEY:
                        reached_keys += 1
                    else:
                        accepted_keys += 1
                        log("key %d accepted d=%.4f steps=%d stalled=%s why=%s" % (i, d, key_steps, stalled, w.get("why")))
                    if "zmin" in w.get("why", "") or "dwell" in w.get("why", ""):
                        # brief hold at a press/dwell point
                        R.step([[0.0, 0.0, 0.0] + self.rot_row(w.get("r")) + [g]] + [[0.0] * 6 + [g]])
                    i += 1
                    key_steps, key_hist = 0, []
                    continue
                key_hist.append(d)
            else:
                d = vnorm(vsub(w["p"], p))
                if i != wd_i:
                    wd_i, wd_best, wd_chunks = i, d, 0
                else:
                    wd_chunks += 1
                    if d < wd_best - 0.003:
                        wd_best, wd_chunks = d, 0
                    elif wd_chunks >= 4:
                        k = i
                        while k < n - 1 and not wps[k]["key"]:
                            k += 1
                        log("watchdog: no progress on wp %d (d=%.4f) -> skip to key %d" % (i, d, k))
                        skips += 1
                        i = k
                        wd_i = -1
                        continue
            # plan a chunk from the measured position, open-loop prediction
            rows = []
            pp = list(p)
            rot = self.rot_row(wps[i].get("r"))
            j = i
            for r in range(self.chunk):
                while j < n and not wps[j]["key"] and vnorm(vsub(wps[j]["p"], pp)) < TOL_WP:
                    j += 1
                if j >= n:
                    break
                e = vsub(wps[j]["p"], pp)
                v = min(V_MAX_CAP, v_of(wps[j]["p"], pp))
                dlt = clip_norm(e, v)
                rows.append(vscale(dlt, 1.0 / POS_SCALE) + (rot if r == 0 else [0.0, 0.0, 0.0]) + [g])
                pp = vadd(pp, dlt)
                if wps[j]["key"] and vnorm(vsub(wps[j]["p"], pp)) < 1e-4:
                    break  # stop at a keypoint; verify by measurement
            if not rows:
                i = j
                continue
            if rem is not None and rem > 0:
                rows = rows[:max(1, rem)]
            # pass-through waypoints are judged on prediction
            while i < j and i < n - 1 and not wps[i]["key"]:
                i += 1
            if wps[i]["key"]:
                key_steps += len(rows)
            R.step(rows)
            self.adapt_chunk(n - i)
            if R.n_steps % 50 < len(rows):
                log("step %d wp %d/%d pos %s tgt %s chunk %d" % (R.n_steps, i, n,
                    [round(x, 4) for x in R.state[:3]], [round(x, 4) for x in wps[min(i, n - 1)]["p"]], self.chunk))
        return {"wp_index": i, "n": n, "reached_keys": reached_keys, "accepted_keys": accepted_keys,
                "skips": skips, "steps": R.n_steps}


def approach_prefix(p_cur, first):
    """Lift-then-traverse prefix from the current pose to the first trajectory point."""
    pre = []
    if vnorm(vsub(first, p_cur)) <= 0.02:
        return pre
    zc = max(p_cur[2], first[2])
    a = [p_cur[0], p_cur[1], zc]
    b = [first[0], first[1], zc]
    if vnorm(vsub(a, p_cur)) > 0.005:
        pre.append({"p": a, "r": None, "key": True, "why": "lift"})
    if vnorm(vsub(b, a)) > 0.005:
        pre.append({"p": b, "r": None, "key": True, "why": "over-first"})
    return pre


def demo_speeds(plan, z_draw):
    """Cap replay speed near the drawing height at ~1.25x the demo's own speed there."""
    pts, times = plan.get("points") or [], plan.get("times") or []
    low_v, high_v = [], []
    for k in range(1, min(len(pts), len(times))):
        dt = times[k] - times[k - 1]
        if dt <= 1e-6:
            continue
        v = vnorm(vsub(pts[k], pts[k - 1])) / (dt * 20.0)   # m per 50 ms control step
        if v < 0.0005:
            continue
        (low_v if min(pts[k][2], pts[k - 1][2]) < z_draw + 0.03 else high_v).append(v)
    v_draw, v_transit = V_DRAW, V_TRANSIT
    if len(low_v) >= 3:
        med = sorted(low_v)[len(low_v) // 2]
        v_draw = max(0.004, min(V_DRAW, 1.25 * med))
    if len(high_v) >= 3:
        med = sorted(high_v)[len(high_v) // 2]
        v_transit = max(0.008, min(0.02, 1.5 * med))
    return v_draw, v_transit


def execute_replay(robot, plan):
    pts = plan["points"]
    use_rot = plan.get("rots") is not None and plan.get("rot_var", 0) >= 0.05
    rots = plan["rots"] if use_rot else None
    wps = build_waypoints(pts, rots)
    z_draw = min(w["p"][2] for w in wps)
    p0 = robot.state[:3]
    pre = approach_prefix(p0, wps[0]["p"])
    for w in pre:
        w["r"] = rots[0] if rots else None
    allw = pre + wps
    save_json("plan_waypoints.json", {"mode": plan["mode"], "use_rot": use_rot, "z_draw": z_draw,
                                       "start": robot.state, "wps": allw})
    log("replay: mode=%s pts=%d wps=%d keys=%d z_draw=%.4f use_rot=%s prefix=%d" % (
        plan["mode"], len(pts), len(wps), sum(1 for w in wps if w["key"]), z_draw, use_rot, len(pre)))
    log("keypoints:", [(k, [round(x, 3) for x in w["p"]], w["why"]) for k, w in enumerate(allw) if w["key"]][:60])
    rem = robot.remaining_steps()
    scale = 1.0
    est = sum(vnorm(vsub(allw[k + 1]["p"], allw[k]["p"])) for k in range(len(allw) - 1)) / V_DRAW
    est += 4 * sum(1 for w in allw if w["key"])
    if rem is not None and est > 0.8 * rem:
        scale = min(2.0, est / (0.8 * rem))
    log("estimated steps %.0f remaining %s speed scale %.2f" % (est, rem, scale))

    v_draw, v_transit = demo_speeds(plan, z_draw)
    log("speeds: draw %.4f transit %.4f m/step (x%.2f)" % (v_draw, v_transit, scale))

    def v_of(tgt, cur):
        low = min(tgt[2], cur[2]) < z_draw + 0.03
        return (v_draw if low else v_transit) * scale

    f = Follower(robot, robot.state[3:6])
    return f.follow(allw, v_of)


def execute_touch(robot, plan):
    """B_touch: visit targets in order, descending until contact."""
    R = robot
    targets = plan["targets"]
    g = gripper_cmd(R.state)
    f = Follower(R, R.state[3:6])
    z_travel = R.state[2]
    z_contact_ref = None
    out = {"touched": []}
    for k, t in enumerate(targets):
        if R.ended or elapsed() > WALL_BUDGET_S:
            break
        hover = [t[0], t[1], z_travel]
        f.follow([{"p": hover, "r": None, "key": True, "why": "hover%d" % k}],
                 lambda a, b: V_TRANSIT)
        # descend until stalled or bound
        z_bound = t[2] - 0.03
        hist = []
        for it in range(60):
            if R.ended:
                break
            p = R.state[:3]
            hist.append(p[2])
            if len(hist) >= 3 and hist[-3] - hist[-1] < 0.002:
                break
            if p[2] < z_bound:
                break
            xy = vsub([t[0], t[1], p[2]], p)
            dlt = clip_norm(xy, 0.004) [:2] + [-0.004]
            R.step([vscale(dlt, 1.0 / POS_SCALE) + f.rot_row(None) + [g]] * 2)
        zc = R.state[2]
        out["touched"].append([round(x, 4) for x in R.state[:3]])
        log("touch %d at %s (target %s)" % (k, [round(x, 4) for x in R.state[:3]], [round(x, 4) for x in t]))
        R.step([[0.0] * 6 + [g]] * 2)
        if z_contact_ref is None:
            z_contact_ref = zc
            z_travel = min(z_travel, zc + 0.06)
        f.follow([{"p": [R.state[0], R.state[1], z_travel], "r": None, "key": True, "why": "up"}],
                 lambda a, b: V_TRANSIT)
    return out


def settle(robot):
    g = gripper_cmd(robot.state)
    for _ in range(2):
        if robot.ended or robot.complete:
            return
        robot.step([[0.0] * 6 + [g]] * (SETTLE_STEPS // 2))


def main(ctx):
    log("controller", VERSION, "start")
    robot = Robot(ctx)
    result = {"version": VERSION}
    try:
        pk = robot.observe()
        result["instruction"] = get_first(pk, "instruction")
        log("instruction:", result["instruction"])
        log("initial state:", robot.state, "step_id:", robot.step_id, "limits:", robot.limits)
        log("packet keys:", list(pk.keys())[:40])
        try:
            txt = json.dumps(pk, default=str)
            save_json("packet0.json", pk if len(txt) < 200000 else {"truncated": txt[:200000]})
        except Exception:
            pass
        try:
            st = reply_json(ctx.call("robomme_status"))
            save_json("status0.json", st)
            robot._flags(st)
            log("status:", str(st)[:1200])
        except Exception as exc:
            log("robomme_status failed", repr(exc)[:300])
        if robot.state is None:
            log("no state; aborting")
            return
        plan = None
        try:
            plan = make_plan(ctx, robot, pk)
        except Exception:
            log("planning crashed", traceback.format_exc()[-1500:])
        result["mode"] = plan["mode"] if plan else None
        log("plan mode:", result["mode"], "elapsed %.1fs" % elapsed())
        if plan is not None and not robot.ended:
            try:
                if plan["mode"] in ("A_state", "A_vision"):
                    result["exec"] = execute_replay(robot, plan)
                elif plan["mode"] == "B_touch":
                    result["exec"] = execute_touch(robot, plan)
            except Exception:
                log("execution crashed", traceback.format_exc()[-1500:])
        if not robot.ended and not robot.complete:
            try:
                settle(robot)
            except Exception:
                log("settle crashed", traceback.format_exc()[-800:])
    except Exception:
        log("main crashed", traceback.format_exc()[-1500:])
    finally:
        result.update({"steps": robot.n_steps, "ended": robot.ended, "task_complete": robot.complete,
                       "final_state": robot.state, "elapsed_s": round(elapsed(), 1), "errors": robot.errors,
                       "mean_call_s": (sum(robot.call_times) / len(robot.call_times)) if robot.call_times else None,
                       "limits": robot.limits})
        log("RESULT", result)
        save_json("result_controller.json", result)
