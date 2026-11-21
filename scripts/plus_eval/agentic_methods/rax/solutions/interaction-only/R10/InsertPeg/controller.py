"""InsertPeg controller v1: track the end-effector/gripper trajectory recorded in the
current episode's conditioning video (frame states), closed loop on measured state.

Rationale: the conditioning video of the current episode shows the robot grasping the
correct peg at the correct end and inserting it into the correct side; its frames carry
the robot state. The execution scene is reset to the video's first frame (verified by
depth comparison, logged at runtime).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
WS = Path("/workspace")
OUT = WS / "output"
POS_SCALE = 0.05
ROT_SCALE = 0.5
BATCH = 5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- rotation helpers
def rv_to_mat(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_rv(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from the diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[i] / max(k[i], 1e-9)
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_delta(cur_rv, tgt_rv):
    """world-frame rotation vector taking cur to tgt."""
    return mat_to_rv(rv_to_mat(tgt_rv) @ rv_to_mat(cur_rv).T)


# ---------------------------------------------------------------- MCP helpers
def meta(result):
    for b in result.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                continue
    return {}


def ended_info(m):
    ended = bool(m.get("episode_ended"))
    complete = False
    for s in (m.get("transition") or {}).get("steps", []) or []:
        ended = ended or bool(s.get("episode_ended"))
        complete = complete or bool(s.get("task_complete"))
    return ended, complete


def load_sequence(m):
    seq = m.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return [], None
    man = json.loads((WS / mp).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    demo = [f for f in frames if f.get("phase") == "video_demo"]
    execf = [f for f in frames if f.get("phase") == "execution"]
    return demo, (execf[-1] if execf else None)


def frame_depth(f, cam="front"):
    for x in f.get("files", []):
        if x["camera"] == cam and x["kind"] == "depth":
            return np.array(Image.open(WS / x["path"])).astype(float)
    return None


# ---------------------------------------------------------------- core
def build_targets(demo):
    tg = []
    for f in demo:
        s = f.get("state")
        if s is None or len(s) < 8:
            continue
        tg.append(np.array(s, float))
    return tg


def gripper_cmds(tg, lookahead=2, open_width=0.075):
    """+1 (close) while the demo's finger opening a few frames ahead is below
    open_width (closing or holding), else -1 (open). Threshold, not derivative:
    the held opening jitters by ~0.1 mm while the peg is moved."""
    n = len(tg)
    cmds = []
    for i in range(n):
        j = min(i + lookahead, n - 1)
        w = tg[j][6] + tg[j][7]
        cmds.append(1.0 if w < open_width else -1.0)
    return cmds


def row_toward(cur, tgt, g):
    dp = (tgt[:3] - cur[:3]) / POS_SCALE
    dr = rot_delta(cur[3:6], tgt[3:6]) / ROT_SCALE
    r = np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [g]])
    return [float(x) for x in r]


def pose_err(cur, tgt):
    return (float(np.linalg.norm(tgt[:3] - cur[:3])),
            float(np.linalg.norm(rot_delta(cur[3:6], tgt[3:6]))))


def step_rows(ctx, rows):
    m = meta(ctx.call("robomme_step", actions=rows))
    return m


def track(ctx, tg, gcmd, state, start=1, rec=None, ff=0.5,
          base_tol=0.004, rot_tol=0.05, max_wait=8, sync_tol=0.003, sync_wait=30):
    """Time-warped closed-loop tracking of tg[start:], one control step per call.
    Advance to the next frame only when within tolerance (or after max_wait steps).
    Before a gripper-command change converge tightly (sync)."""
    n = len(tg)
    k = start
    waited = 0
    steps_total = 0
    state = np.array(state, float)
    while k < n:
        t = tg[k]
        prev = tg[k - 1]
        g = gcmd[k]
        is_sync = gcmd[k] != gcmd[k - 1]
        # feedforward: demo velocity at this frame
        vel = t[:3] - prev[:3]
        dp = (t[:3] - state[:3] + ff * vel) / POS_SCALE
        dr = rot_delta(state[3:6], t[3:6]) / ROT_SCALE
        if is_sync:
            g = gcmd[k - 1]  # do not switch the gripper until aligned
        row = [float(x) for x in np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [g]])]
        m = step_rows(ctx, [row])
        steps_total += 1
        if "state" in m:
            state = np.array(m["state"], float)
        ended, complete = ended_info(m)
        if ended:
            log("episode ended at k", k, "complete", complete)
            return state, True, complete
        rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
        pe, re_ = pose_err(state, t)
        speed = float(np.linalg.norm(vel))
        if is_sync:
            ok = (pe < sync_tol and re_ < rot_tol) or waited >= sync_wait
        else:
            ok = (pe < base_tol + 2 * speed and re_ < rot_tol + 2 * float(np.linalg.norm(rot_delta(prev[3:6], t[3:6])))) or waited >= max_wait
        if rem is not None and rem < (n - k) + 5:
            ok = True  # running out of steps: stop waiting
        if ok:
            if is_sync:
                log("sync at k", k, "err", round(pe, 4), round(re_, 3), "waited", waited, "z", round(float(state[2]), 4))
                # now actually switch the gripper and let it act for a few steps on the spot
                hold = [float(x) for x in np.concatenate([np.clip((t[:3] - state[:3]) / POS_SCALE, -1, 1),
                                                          np.clip(rot_delta(state[3:6], t[3:6]) / ROT_SCALE, -1, 1),
                                                          [gcmd[k]]])]
                m = step_rows(ctx, [hold])
                if "state" in m:
                    state = np.array(m["state"], float)
            if rec is not None:
                rec.append({"k": k, "step_id": m.get("step_id"), "err": round(pe, 4), "rerr": round(re_, 3),
                            "waited": waited, "state": [round(float(v), 4) for v in state]})
            k += 1
            waited = 0
        else:
            waited += 1
    return state, False, False


# ---------------------------------------------------------------- insertion
def insertion_split(tg, gcmd, pre=0.045):
    """Find the demo's final straight insertion segment. Returns (k_switch, p0, axis, L, R_rv)
    or None. axis is the unit direction of the final straight motion; k_switch the first
    frame whose remaining axial distance to the final pose is <= pre."""
    n = len(tg)
    final = tg[-1][:3]
    back = None
    for j in range(n - 1, -1, -1):
        if np.linalg.norm(tg[j][:3] - final) > 0.03:
            back = j
            break
    if back is None or gcmd[-1] < 0:
        return None
    axis = final - tg[back][:3]
    axis = axis / np.linalg.norm(axis)
    k_sw = None
    for j in range(back, -1, -1):
        rel = final - tg[j][:3]
        along = float(rel @ axis)
        perp = float(np.linalg.norm(rel - along * axis))
        if perp > 0.004 or gcmd[j] < 0:
            break
        if along <= pre:
            k_sw = j
    if k_sw is None:
        return None
    p0 = tg[k_sw][:3].copy()
    L = float((final - p0) @ axis)
    return k_sw, p0, axis, L, tg[-1][3:6].copy()


def spiral(rings=((0.3, 8), (0.6, 8), (0.9, 12), (1.2, 12), (1.6, 16), (2.0, 16), (2.5, 16))):
    pts = [(0.0, 0.0)]
    for r, n in rings:
        for i in range(n):
            th = 2 * np.pi * i / n
            pts.append((r * np.cos(th), r * np.sin(th)))
    return pts


def search_list():
    """(vertical mm, lateral mm, yaw rad) candidates, finest first."""
    S = [(0.0, 0.0, 0.0)]
    for d in (0.25, 0.5, 0.75, 1.0):
        S += [(0.0, d, 0.0), (0.0, -d, 0.0)]
    for d in (0.25, 0.5):
        S += [(d, 0.0, 0.0), (-d, 0.0, 0.0)]
    for y in (0.004, 0.008):
        S += [(0.0, 0.0, y), (0.0, 0.0, -y)]
    for b in (0.5, -0.5):
        for y in (0.006, -0.006):
            S.append((0.0, b, y))
    for oa, ob in spiral(((0.6, 8), (1.25, 12), (1.6, 12), (2.0, 16))):
        if (oa, ob) != (0.0, 0.0):
            S.append((oa, ob, 0.0))
    return S


SEARCH = search_list()


def rot_z(rv, yaw):
    if yaw == 0:
        return rv
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return mat_to_rv(Rz @ rv_to_mat(rv))


def insert(ctx, state, p0, axis, L, rv, extra=0.006, push_rate=0.001, lead=0.002,
           retreat=0.003, reserve=3, info=None):
    """Precise insertion along the demo's straight segment.
    Perpendicular position held with P+I control (removes ~0.3 mm steady bias);
    on a stall (no axial progress while commanded ahead) retreat `retreat` and try the
    next fine spiral offset (mm in (vertical u, lateral w))."""
    up = np.array([0.0, 0.0, 1.0])
    u = up - (up @ axis) * axis
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(axis, u)
    goal = L + extra
    integ = np.zeros(3)
    attempt = 0
    mode = "align"
    s_contact = None
    entered = False
    best = -1.0
    n_stall = 0
    n_deep = 0
    align_steps = 0
    s_cmd = max(0.0, float((state[:3] - p0) @ axis))
    s_align = s_cmd
    log("insert start: L", round(L, 4), "s", round(s_cmd, 4))
    while True:
        oa, ob, yaw = SEARCH[attempt % len(SEARCH)]
        off = (oa * u + ob * w) * 0.001
        rv_t = rot_z(rv, yaw)
        s_meas = float((state[:3] - p0) @ axis)
        best = max(best, s_meas)
        if mode == "align":
            s_tgt = s_align
        else:
            s_cmd = min(goal, max(s_cmd, s_meas) + push_rate)
            s_tgt = min(s_cmd, s_meas + lead)
        des = p0 + axis * s_tgt + off
        err = des - state[:3]
        perp_err = err - (err @ axis) * axis
        integ = np.clip(integ + 0.4 * perp_err, -0.002, 0.002)
        dp = np.clip((err + integ) / POS_SCALE, -1, 1)
        dr = np.clip(rot_delta(state[3:6], rv_t) / ROT_SCALE, -1, 1)
        row = [float(x) for x in np.concatenate([dp, dr, [1.0]])]
        m = step_rows(ctx, [row])
        if "state" in m:
            state = np.array(m["state"], float)
        ended, complete = ended_info(m)
        if ended:
            log("insert: episode ended complete", complete, "s", round(s_meas, 4), "attempt", attempt)
            return state, True, complete
        rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
        if rem is not None and rem <= reserve:
            log("insert: out of steps; best s", round(best, 4), "of", round(L, 4), "attempts", attempt)
            return state, False, False
        s_new = float((state[:3] - p0) @ axis)
        e2 = des - state[:3]
        pe = float(np.linalg.norm(e2 - (e2 @ axis) * axis))
        if mode == "align":
            align_steps += 1
            if (pe < 0.00012 and abs(s_new - s_align) < 0.0006
                    and pose_err(state, np.concatenate([des, rv_t]))[1] < 0.0015) or align_steps > 10:
                mode = "push"
                s_cmd = s_new
                n_stall = 0
            continue
        # push mode
        if s_new >= goal - 0.001:
            log("insert: reached goal s", round(s_new, 4), "attempt", attempt, "off", (round(oa, 2), round(ob, 2), yaw))
            if info is not None:
                info["reached"] = True
            return state, False, False
        if s_contact is not None and not entered and s_new > s_contact + 0.004:
            entered = True
            log("insert: passed contact depth; s", round(s_new, 4), "attempt", attempt, "off", (round(oa, 2), round(ob, 2), yaw))
        if s_new - s_meas < 0.00025 and s_tgt - s_new > 0.0008:
            n_stall += 1
        else:
            n_stall = 0
        if n_stall >= 2:
            if s_contact is None:
                s_contact = s_new
            if attempt < 3 or attempt % 10 == 0:
                log("insert: stall s", round(s_new, 4), "attempt", attempt, "off", (round(oa, 2), round(ob, 2), yaw),
                    "perr_mm", round(pe * 1000, 2), "rerr", round(pose_err(state, np.concatenate([des, rv_t]))[1], 4))
            if entered and s_new > s_contact + 0.004 and n_deep < 4:
                # stuck deeper inside the hole: small back-off, same offset
                n_deep += 1
                log("insert: deep stall s", round(s_new, 4), "n", n_deep)
                s_align = s_new - 0.0015
                mode = "align"
                align_steps = 0
                n_stall = 0
                continue
            n_deep = 0
            entered = False
            attempt += 1
            s_align = max(-0.01, min(s_new, s_contact) - retreat)
            mode = "align"
            align_steps = 0
            n_stall = 0
            integ[:] = integ  # keep the integral (bias is pose-dependent, not offset-dependent)


def compliant_push(ctx, state, p0, axis, goal, s_start, max_steps=60, lead0=0.003, reserve=3):
    """Push along `axis` commanding only an axial delta (measured pose + delta), so the
    perpendicular position and the orientation are free to comply with the hole.
    Escalate lead and add small yaw/lateral wiggles while stalled.
    Returns (state, ended, complete, s_reached, steps_used)."""
    up = np.array([0.0, 0.0, 1.0])
    u = up - (up @ axis) * axis
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(axis, u)
    lead = lead0
    stall = 0
    steps = 0
    s_prev = float((state[:3] - p0) @ axis)
    best = s_prev
    wig = 0
    while steps < max_steps:
        dp = axis * lead
        dr = np.zeros(3)
        if stall >= 3:
            # wiggle: alternate small yaw and lateral pulses
            ph = wig % 4
            yaw = 0.01 * (1 if ph in (0, 1) else -1) * (1 if ph % 2 == 0 else -1)
            dr = np.array([0.0, 0.0, yaw])
            dp = dp + w * (0.0006 if ph < 2 else -0.0006) + u * (0.0004 if ph % 2 == 0 else -0.0004)
            wig += 1
        row = [float(x) for x in np.concatenate([np.clip(dp / POS_SCALE, -1, 1), np.clip(dr / ROT_SCALE, -1, 1), [1.0]])]
        m = step_rows(ctx, [row])
        steps += 1
        if "state" in m:
            state = np.array(m["state"], float)
        ended, complete = ended_info(m)
        s_now = float((state[:3] - p0) @ axis)
        best = max(best, s_now)
        if ended:
            return state, True, complete, best, steps
        rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
        if rem is not None and rem <= reserve:
            return state, False, False, best, steps
        if s_now >= goal - 0.001:
            return state, False, False, best, steps
        if s_now - s_prev < 0.0003:
            stall += 1
            if stall in (6, 12, 18):
                lead = min(0.008, lead + 0.0015)
        else:
            stall = 0
            lead = lead0
        if stall > 30:
            break
        s_prev = s_now
    return state, False, False, best, steps


def insert_v6(ctx, state, p0, axis, L, rv, extra=0.006, pre=0.003, reserve=3, info=None, max_attempts=40):
    """Stiff alignment to the demo pose up to `pre` along the axis (tip still outside the
    face), then compliant axial pushing; if that stalls, retreat, re-align with the next
    search offset and push compliantly again."""
    up = np.array([0.0, 0.0, 1.0])
    u = up - (up @ axis) * axis
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(axis, u)
    goal = L + extra
    integ = np.zeros(3)
    attempt = 0
    best = -1.0
    while True:
        oa, ob, yaw = SEARCH[attempt % len(SEARCH)]
        off = (oa * u + ob * w) * 0.001
        rv_t = rot_z(rv, yaw)
        # stiff align at s = pre
        for i in range(14):
            des = p0 + axis * pre + off
            err = des - state[:3]
            integ = np.clip(integ + 0.4 * (err - (err @ axis) * axis), -0.002, 0.002)
            dp = np.clip((err + integ) / POS_SCALE, -1, 1)
            dr = np.clip(rot_delta(state[3:6], rv_t) / ROT_SCALE, -1, 1)
            m = step_rows(ctx, [[float(x) for x in np.concatenate([dp, dr, [1.0]])]])
            if "state" in m:
                state = np.array(m["state"], float)
            ended, complete = ended_info(m)
            if ended:
                return state, True, complete
            rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
            if rem is not None and rem <= reserve:
                return state, False, False
            pe, re_ = pose_err(state, np.concatenate([des, rv_t]))
            if pe < 0.00015 and re_ < 0.0015:
                break
        state, ended, complete, s_best, used = compliant_push(ctx, state, p0, axis, goal, pre)
        best = max(best, s_best)
        log("insert_v6 attempt", attempt, "off", (round(oa, 2), round(ob, 2), yaw), "reached s", round(s_best, 4),
            "of", round(goal, 4), "steps", used, "ended", ended, "complete", complete)
        if ended:
            return state, True, complete
        if s_best >= goal - 0.001:
            if info is not None:
                info["reached"] = True
            return state, False, False
        m = meta(ctx.call("robomme_status")) if False else None
        attempt += 1
        if attempt >= max_attempts:
            return state, False, False


def result_depth(result, idx=4):
    """Decode the wrist depth image (content index 4) of an MCP result into mm array."""
    import base64, io
    try:
        content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
        b = content[idx]
        data = b.get("data") if isinstance(b, dict) else getattr(b, "data", None)
        return np.array(Image.open(io.BytesIO(base64.b64decode(data)))).astype(float)
    except Exception as e:
        log("result_depth failed", repr(e))
        return None


def hole_shift(ctx, demo, tg, k, p0, axis, max_shift=0.014):
    """Observe at the pre-insertion pose and estimate the hole's perpendicular world shift
    relative to the demo by registering wrist-depth hole regions. Returns (shift, info)."""
    import holeloc as H
    res = ctx.call("robomme_observe")
    m = meta(res)
    st = np.array(m["state"], float)
    dL = result_depth(res)
    errs = [pose_err(st, x)[0] + 0.05 * pose_err(st, x)[1] for x in tg]
    j = int(np.argmin(errs))
    def fpath(f, kind):
        for x in f.get("files", []):
            if x.get("camera") == "wrist" and x.get("kind") == kind:
                return WS / x["path"]
    try:
        K = m["camera_intrinsics"]["wrist"]["matrix"]
        dD = np.array(Image.open(fpath(demo[j], "depth"))).astype(float)
        dC = np.array(Image.open(fpath(demo[int(k * 0.75)], "depth"))).astype(float)
        peg = (dC > 0) & (dC < 75)
        r = H.register_3d(dL, m["camera_poses"]["wrist"], dD, demo[j]["camera_poses"]["wrist"], K, p0, axis, peg,
                          max_shift=max_shift)
    except Exception as e:
        log("hole_shift failed", repr(e))
        return np.zeros(3), {"error": repr(e)}
    if r is None:
        log("hole_shift: no registration")
        return np.zeros(3), {"frame": j}
    info = {"frame": j, "pose_err": pose_err(st, tg[j])[0], "shift": [float(x) for x in r[0]],
            "score": r[1], "A": r[2], "B": r[3], "n": r[4:]}
    log("hole_shift", info)
    if r[1] < 0.4:
        return np.zeros(3), info
    return r[0], info


def stiff_move(ctx, state, des, rv, n=25, tol=0.00015, reserve=3):
    integ = np.zeros(3)
    m = {}
    for i in range(n):
        err = des - state[:3]
        integ = np.clip(integ + 0.4 * err, -0.002, 0.002)
        dp = np.clip((err + integ) / POS_SCALE, -1, 1)
        dr = np.clip(rot_delta(state[3:6], rv) / ROT_SCALE, -1, 1)
        m = step_rows(ctx, [[float(x) for x in np.concatenate([dp, dr, [1.0]])]])
        if "state" in m:
            state = np.array(m["state"], float)
        ended, complete = ended_info(m)
        if ended:
            return state, True, complete
        rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
        if rem is not None and rem <= reserve:
            return state, False, False
        pe, re_ = pose_err(state, np.concatenate([des, rv]))
        if pe < tol and re_ < 0.0015:
            break
    return state, False, False


def grid_offsets(pitch=1.5, rmax=6.1):
    pts = []
    n = int(rmax / pitch) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            a, b = i * pitch, j * pitch
            r = (a * a + b * b) ** 0.5
            if r <= rmax + 1e-9:
                pts.append((r, abs(b), a, b))
    pts.sort()
    return [(a, b) for _, _, a, b in pts]


def grid_offsets_aniso(pitch=1.0, umax=3.0, vmax=7.0, su=2.5, sv=6.0):
    """(u, v) mm offsets ordered by normalised distance; the vision estimate is better
    laterally (u) than vertically (v)."""
    pts = []
    nu = int(round(umax / pitch)); nv = int(round(vmax / pitch))
    for i in range(-nu, nu + 1):
        for j in range(-nv, nv + 1):
            a, b = i * pitch, j * pitch
            pts.append((((a / su) ** 2 + (b / sv) ** 2) ** 0.5, abs(b), abs(a), a, b))
    pts.sort()
    return [(p[3], p[4]) for p in pts]


GRID = grid_offsets_aniso(umax=2.0, vmax=6.0, su=2.0, sv=5.0)


def _row(dp, dr, g=1.0):
    return [float(x) for x in np.concatenate([np.clip(dp / POS_SCALE, -1, 1), np.clip(dr / ROT_SCALE, -1, 1), [g]])]


def insert_grid(ctx, state, p0, axis, L, rv, extra=0.008, pre=0.006, lead=0.004, enter_margin=0.003,
                reserve=3, info=None):
    """Probe a perpendicular grid of offsets around p0 (estimated hole line): stiff align at s=pre,
    short stiff push; if the peg passes the face depth by `enter_margin`, finish compliantly."""
    import holeloc as H
    _, e1, e2 = H.perp_basis(axis)
    goal = L + extra
    s_face = None
    tries = []
    for idx, (oa, ob) in enumerate(GRID):
        off = (oa * e1 + ob * e2) * 0.001
        line = p0 + off
        # retreat/align at s=pre (stiff)
        pre_eff = pre if s_face is None else max(0.0, min(pre, s_face - 0.002))
        state, ended, complete = stiff_move(ctx, state, line + axis * pre_eff, rv, n=10, tol=0.0004, reserve=reserve)
        if ended:
            return state, True, complete
        rem_ok = True
        # short stiff push
        s_prev = float((state[:3] - line) @ axis)
        best = s_prev
        stall = 0
        for i in range(40):
            s_cur = float((state[:3] - line) @ axis)
            des = line + axis * min(s_cur + lead, goal)
            err = des - state[:3]
            dr = rot_delta(state[3:6], rv)
            m = step_rows(ctx, [_row(err, dr)])
            if "state" in m:
                state = np.array(m["state"], float)
            ended, complete = ended_info(m)
            if ended:
                return state, True, complete
            rem = (m.get("episode_limits") or {}).get("remaining_episode_steps")
            if rem is not None and rem <= reserve:
                rem_ok = False
                break
            s_now = float((state[:3] - line) @ axis)
            best = max(best, s_now)
            if s_now > ((s_face + enter_margin) if s_face is not None else 0.0135):
                break
            if s_now - s_prev < 0.0004:
                stall += 1
                near_face = s_face is not None and s_now > s_face - 0.001
                if stall >= 2 or near_face:
                    break
            else:
                stall = 0
            s_prev = s_now
        tries.append((oa, ob, round(best * 1000, 1)))
        log("grid try", idx, "off mm", (oa, ob), "best s mm", round(best * 1000, 1), "s_face", None if s_face is None else round(s_face * 1000, 1))
        if not rem_ok:
            break
        thr = (s_face + enter_margin) if s_face is not None else 0.0135
        if best <= thr and s_face is None:
            s_face = best
            continue
        if best > thr:
            log("entered hole at offset", (oa, ob))
            if info is not None:
                info["entered"] = (oa, ob, best)
            state, ended, complete, s_best, used = compliant_push(ctx, state, line, axis, goal, best, max_steps=150)
            log("final push reached", round(s_best * 1000, 1), "of", round(goal * 1000, 1), "ended", ended, "complete", complete)
            if ended:
                return state, True, complete
            # hold pushing a little more, then stop
            return state, False, False
        if best < s_face - 0.002:
            # earlier contact than the face: note but continue
            pass
        s_face = min(s_face, best) if best > s_face - 0.003 else s_face
    if info is not None:
        info["grid_tries"] = tries
    return state, False, False


def _main(ctx):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    m = meta(ctx.call("robomme_observe"))
    log("instruction:", m.get("instruction"), "step", m.get("step_id"))
    state = np.array(m["state"], float)
    demo, execf = load_sequence(m)
    log("demo frames", len(demo))
    if execf is not None and demo:
        d0, d1 = frame_depth(demo[0]), frame_depth(execf)
        if d0 is not None and d1 is not None:
            log("layout check: front depth px differing >5mm:", int((np.abs(d0 - d1) > 5).sum()))
    tg = build_targets(demo)
    if len(tg) < 10:
        log("no usable demo states; abort")
        return
    gcmd = gripper_cmds(tg)
    rec = []
    split = insertion_split(tg, gcmd)
    info = {}
    if split is None:
        log("no insertion segment found; plain tracking")
        state, ended, complete = track(ctx, tg, gcmd, state, start=1, rec=rec)
    else:
        k_sw, p0, axis, L, rv = split
        log("insertion split at k", k_sw, "of", len(tg), "axis", axis.round(3).tolist(), "L", round(L, 4))
        state, ended, complete = track(ctx, tg[:k_sw + 1], gcmd[:k_sw + 1], state, start=1, rec=rec)
        log("tracking done; ended", ended, "t", round(time.time() - t0, 1))
        sh = np.zeros(3)
        if not ended:
            # locate the hole relative to the demo (the hole offset can differ from the video)
            state, ended, complete = stiff_move(ctx, state, p0.copy(), rv)
        if not ended:
            sh1, i1 = hole_shift(ctx, demo, tg, k_sw, p0, axis)
            info["hole1"] = i1
            sh = sh1
            if np.linalg.norm(sh1) > 0.0015:
                state, ended, complete = stiff_move(ctx, state, p0 + sh1, rv)
                if not ended:
                    sh2, i2 = hole_shift(ctx, demo, tg, k_sw, p0, axis)
                    info["hole2"] = i2
                    if i2.get("A", 0) >= 0.8 and np.linalg.norm(sh2 - sh1) < 0.006:
                        sh = sh2
            log("hole shift used (mm)", [round(float(v) * 1000, 2) for v in sh])
        if not ended:
            state, ended, complete = insert_grid(ctx, state, p0 + sh, axis, L, rv, info=info)
    log("after insert: ended", ended, "complete", complete, "t", round(time.time() - t0, 1))
    if not ended:
        # hold still with the gripper closed so the simulator can register the state
        for _ in range(3):
            mm = step_rows(ctx, [[0, 0, 0, 0, 0, 0, 1.0]] * 3)
            ended, complete = ended_info(mm)
            if ended:
                break
        log("hold; ended", ended, "complete", complete, "state", [round(float(v), 4) for v in state])
    (OUT / "track_log.json").write_text(json.dumps(rec))
    try:
        (OUT / "insert_info.json").write_text(json.dumps(info, default=str))
    except Exception as e:
        log("info write failed", repr(e))
    log("done", round(time.time() - t0, 1), "s")


def main(ctx):
    try:
        _main(ctx)
    except Exception:
        import traceback
        log("controller exception:", traceback.format_exc())
