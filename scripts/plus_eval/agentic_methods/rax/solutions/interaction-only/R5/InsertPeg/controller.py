"""InsertPeg controller v1: imitate the conditioning video's robot trajectory.

The conditioning video of the current episode carries the demonstrating robot's
measured state at every frame. The execution scene starts from the video's first
layout (checked via depth difference). We servo the end effector through the
video's state sequence (closed loop on measured state) with the same gripper
timing, then finish the insertion.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

WS = Path("/workspace")
OUT = WS / "output"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ---------------- rotation helpers (numpy only) ----------------
def rotvec_to_R(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_err(target_rv, cur_rv):
    """world-frame rotation vector taking current to target"""
    return R_to_rotvec(rotvec_to_R(target_rv) @ rotvec_to_R(cur_rv).T)


# ---------------- MCP helpers ----------------
def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ended = False
        self.complete = False
        self.state = None
        self.step_id = 0
        self.remaining = None
        self.last = None

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._update(p)
        return p

    def _update(self, p):
        if "state" in p:
            self.state = np.array(p["state"], float)
        self.step_id = p.get("step_id", self.step_id)
        lim = p.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if p.get("episode_ended"):
            self.ended = True
        tr = p.get("transition") or {}
        for s in tr.get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if p.get("task_complete"):
            self.complete = True
        self.last = p

    def step(self, rows):
        if self.ended:
            return None
        if self.remaining is not None:
            rows = rows[: max(0, self.remaining)]
        if not rows:
            return None
        rows = [[float(np.clip(x, -1, 1)) for x in r] for r in rows]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        self._update(p)
        return p


def servo_row(state, tpos, trot, grip, pos_gain=1.0, rot_gain=1.0, max_pos=1.0, max_rot=1.0):
    dp = (np.asarray(tpos) - state[:3]) / 0.05 * pos_gain
    n = np.max(np.abs(dp))
    if n > max_pos:
        dp = dp / n * max_pos
    dr = rot_err(trot, state[3:6]) / 0.5 * rot_gain
    n = np.max(np.abs(dr))
    if n > max_rot:
        dr = dr / n * max_rot
    return list(dp) + list(dr) + [grip]


# ---------------- video parsing ----------------
def load_video(obs):
    seq = obs.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if mp:
        mpath = WS / mp
    else:  # fallback: newest step-0 manifest among this container's run frames
        cands = sorted((WS / "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"),
                       key=lambda q: q.stat().st_mtime)
        if not cands:
            return None, None
        mpath = cands[-1]
        log("manifest fallback", str(mpath))
    man = json.loads(mpath.read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") == "execution"]
    return vid, (exe[-1] if exe else None)


def depth_of(frame, cam):
    from PIL import Image
    f = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    with Image.open(WS / f[(cam, "depth")]) as im:
        return np.array(im).astype(float) * 0.001


def main(ctx):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rob = Robot(ctx)
    try:
        run(rob, t0)
    except Exception as e:  # never crash
        import traceback
        log("EXCEPTION", repr(e), traceback.format_exc())
    finally:
        try:
            (OUT / "controller_log.txt").write_text("\n".join(LOG))
        except Exception:
            pass



def find_insert_start(S, close_idx):
    """first index after which z stays within 3 mm of the final z (insertion height), after grasp."""
    zf = S[-1, 2]
    ok = np.abs(S[:, 2] - zf) < 0.003
    k = len(S) - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    if close_idx is not None and k <= close_idx + 20:
        k = max(close_idx + 20, len(S) - 100)
    return min(k, len(S) - 2)


def path_param(P, p):
    """project p onto polyline P (Nx3); return arc length s, segment idx"""
    best = (1e9, 0.0, 0)
    cum = 0.0
    for j in range(len(P) - 1):
        a, b = P[j], P[j + 1]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-9:
            continue
        t = np.clip(np.dot(p - a, ab) / L ** 2, 0, 1)
        d = np.linalg.norm(a + t * ab - p)
        if d < best[0]:
            best = (d, cum + t * L, j)
        cum += L
    return best[1], best[2], best[0]


def point_at(P, Rv, s):
    cum = 0.0
    for j in range(len(P) - 1):
        L = np.linalg.norm(P[j + 1] - P[j])
        if L < 1e-9:
            continue
        if cum + L >= s:
            t = (s - cum) / L
            return P[j] + t * (P[j + 1] - P[j]), Rv[j + 1]
        cum += L
    # extrapolate along the final direction
    d = P[-1] - P[max(0, len(P) - 8)]
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.zeros(3)
    return P[-1] + d * (s - cum), Rv[-1]


def yaw_rot(rv, dyaw):
    c, s = np.cos(dyaw), np.sin(dyaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R_to_rotvec(Rz @ rotvec_to_R(rv))


def insert_phase(rob, P, Rv, grip, t0, extend=0.02, look=0.010, max_steps=300):
    # dedupe path points
    keep = [0]
    for j in range(1, len(P)):
        if np.linalg.norm(P[j] - P[keep[-1]]) > 0.0015:
            keep.append(j)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    P = P[keep]; Rv = Rv[keep]
    total = sum(np.linalg.norm(P[j + 1] - P[j]) for j in range(len(P) - 1))
    s_goal = total + extend
    log("insert: path pts %d length %.3f goal %.3f" % (len(P), total, s_goal))
    best_s = -1; last_prog_step = rob.step_id; wiggle = 0; n = 0; retries = 0
    offsets = [np.zeros(3), np.array([0, 0, 0.002]), np.array([0, 0, -0.002])]
    off = np.zeros(3)
    while not rob.ended and n < max_steps:
        s, j, dist = path_param(P, rob.state[:3] - off)
        if s > best_s + 0.001:
            best_s = s; last_prog_step = rob.step_id
        if s >= s_goal - 0.001:
            log("insert: reached goal s=%.3f" % s)
            break
        stalled = rob.step_id - last_prog_step
        lk = look
        dyaw = 0.0
        if stalled > 6:
            lk = look * 2.5
            wiggle += 1
            dyaw = 0.05 * (1 if (wiggle // 2) % 2 == 0 else -1)
        if stalled > 40:
            # back off and retry with a small offset
            retries += 1
            log("insert: stalled at s=%.3f (step %d), retry %d" % (s, rob.step_id, retries))
            if retries > 4:
                break
            off = offsets[retries % len(offsets)]
            sb = max(0.0, s - 0.015)
            for _ in range(12):
                tp, tr = point_at(P, Rv, sb)
                rob.step([servo_row(rob.state, tp + off, tr, grip, max_pos=0.3)])
            best_s = sb; last_prog_step = rob.step_id
            continue
        tp, tr = point_at(P, Rv, min(s + lk, s_goal))
        if dyaw:
            tr = yaw_rot(tr, dyaw)
        rob.step([servo_row(rob.state, tp + off, tr, grip, max_pos=0.4)])
        n += 1
        if n % 10 == 0:
            log("  ins n %d step %d s %.3f/%.3f dist %.4f stalled %d pos %s w %.4f" % (
                n, rob.step_id, s, s_goal, dist, stalled, np.round(rob.state[:3], 4).tolist(),
                rob.state[6] + rob.state[7]))
    log("insert end: step %d s_best %.3f complete %s ended %s" % (rob.step_id, best_s, rob.complete, rob.ended))



def frame_depth(frame, cam):
    return depth_of(frame, cam)


def obs_frame(rob):
    """latest observation as a frame-like dict with depth arrays (via saved files)."""
    p = rob.observe()
    seq = p.get("frame_sequence") or {}
    return p


def latest_files(p):
    seq = p.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None
    man = json.loads((WS / mp).read_text())
    fr = sorted(man["frames"], key=lambda f: f["frame_index"])[-1]
    return fr


def converge(rob, tpos, trot, grip, tol=0.0007, rtol=0.004, n=25, max_pos=0.3):
    for _ in range(n):
        if rob.ended:
            return
        e = np.linalg.norm(rob.state[:3] - tpos)
        r = np.linalg.norm(rot_err(trot, rob.state[3:6]))
        if e < tol and r < rtol:
            return
        rob.step([servo_row(rob.state, tpos, trot, grip, max_pos=max_pos)])


def reply_depth(reply, cam):
    """decode the depth attachment of camera `cam` from an MCP reply"""
    import base64, io
    from PIL import Image
    p = packet(reply)
    for a in p.get("attachments", []):
        if a.get("camera") == cam and a.get("kind") == "depth":
            b = reply["content"][a["content_index"]]
            with Image.open(io.BytesIO(base64.b64decode(b["data"]))) as im:
                return np.array(im).astype(float) * 0.001, p
    return None, p


def measure_peg_correction(rob, vid, S, k_ref, grip):
    """Tool-frame correction (R_d, t_d) with X_video = R_d X_ours + t_d for the in-hand peg."""
    try:
        from perception import peg_line_tool, peg_correction
        converge(rob, S[k_ref, :3], S[k_ref, 3:6], grip, tol=0.0005, rtol=0.003, n=20)
        for _ in range(3):  # settle (stationary like the reference frame)
            rob.step([servo_row(rob.state, S[k_ref, :3], S[k_ref, 3:6], grip)])
        reply = rob.ctx.call("robomme_observe")
        dcur, p = reply_depth(reply, "wrist")
        rob._update(p)
        if dcur is None:
            log("peg correction: no wrist depth")
            return np.eye(3), np.zeros(3)
        dref = depth_of(vid[k_ref], "wrist")
        lv = peg_line_tool(dref, vid[k_ref]["camera_intrinsics"]["wrist"], vid[k_ref]["camera_poses"]["wrist"], vid[k_ref]["state"])
        lo = peg_line_tool(dcur, p["camera_intrinsics"]["wrist"], p["camera_poses"]["wrist"], p["state"])
        if lv is None or lo is None:
            log("peg correction: line fit failed", lv is None, lo is None)
            return np.eye(3), np.zeros(3)
        raw_ang = np.degrees(np.arccos(np.clip(np.dot(lv[1], lo[1]), -1, 1)))
        Rd, td, ang, ax = peg_correction(lv[:2], lo[:2], max_rot_deg=10.0, max_trans=0.05)
        if raw_ang > 2.5 or np.linalg.norm(td) > 0.0025 or min(lv[2], lo[2]) < 40:
            log("peg correction REJECTED (implausible): rot %.2f t mm %s n %d/%d" % (
                raw_ang, np.round(td * 1000, 2).tolist(), lv[2], lo[2]))
            return np.eye(3), np.zeros(3)
        log("peg line video m %s e %s n %d | ours m %s e %s n %d" % (
            np.round(lv[0] * 1000, 2).tolist(), np.round(lv[1], 4).tolist(), lv[2],
            np.round(lo[0] * 1000, 2).tolist(), np.round(lo[1], 4).tolist(), lo[2]))
        log("peg correction: rot %.2f deg axis %s, t mm %s" % (ang, np.round(ax, 2).tolist(), np.round(td * 1000, 2).tolist()))
        return Rd, td
    except Exception as e:
        import traceback
        log("peg correction failed", repr(e), traceback.format_exc())
        return np.eye(3), np.zeros(3)


def apply_corr(tp, tr, corr):
    """video tool target (tp, rotvec tr) -> commanded pose so that our peg matches the video's."""
    Rd, td = corr
    Rv = rotvec_to_R(tr)
    return tp + Rv @ td, R_to_rotvec(Rv @ Rd)


def insert_v4(rob, P, Rv, grip, corr, t0, step_len=0.0012, max_total_steps=430):
    """P,Rv: video insertion path (pre-contact -> final). corr: in-hand peg correction (R_d, t_d)."""
    keep = [0]
    for j in range(1, len(P)):
        if np.linalg.norm(P[j] - P[keep[-1]]) > 0.001:
            keep.append(j)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    P = P[keep]; Rv = Rv[keep]
    total = sum(np.linalg.norm(P[j + 1] - P[j]) for j in range(len(P) - 1))
    s_goal = total + 0.012
    dfin = P[-1] - P[max(0, len(P) - 6)]
    dfin = dfin / max(np.linalg.norm(dfin), 1e-9)
    lat = np.cross([0, 0, 1.0], dfin); lat /= max(np.linalg.norm(lat), 1e-9)
    up = np.array([0, 0, 1.0])
    g = 0.0007
    rg = np.radians(0.5)
    # (lateral, vertical, pitch about lat, yaw about z)
    pattern = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1),
               (0, 1, 0, 0), (0, -1, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0),
               (0, 0, 2, 0), (0, 0, -2, 0), (0, 0, 0, 2), (0, 0, 0, -2),
               (0, 1, 1, 0), (0, -1, -1, 0), (0, 1, -1, 0), (0, -1, 1, 0),
               (1, 0, 0, 1), (-1, 0, 0, -1), (1, 0, 0, -1), (-1, 0, 0, 1),
               (0, 2, 0, 0), (0, -2, 0, 0), (2, 0, 0, 0), (-2, 0, 0, 0)]
    log("insert v4: path pts %d length %.3f" % (len(P), total))
    start_step = rob.step_id
    best_s = 0.0
    attempt = 0
    s_contact = None

    def pose(s, a=0, b=0, c=0, d=0):
        tp, tr = point_at(P, Rv, s)
        pc, rc = apply_corr(tp, tr, corr)
        pc = pc + lat * a * g + up * b * g
        if c or d:
            dR = rotvec_to_R(lat * c * rg) @ rotvec_to_R(up * d * rg)
            rc = R_to_rotvec(dR @ rotvec_to_R(rc))
        return pc, rc, pc - tp  # commanded pos, rot, positional shift vs video path

    def progress(shift):
        q = rob.state[:3] - shift
        s_meas, j, dist = path_param(P, q)
        if s_meas > total:
            s_meas = total + np.dot(q - P[-1], dfin)
        return s_meas

    while not rob.ended and attempt < len(pattern) and rob.step_id - start_step < max_total_steps:
        a, b, c, d = pattern[attempt]
        s0 = 0.0 if s_contact is None else max(0.0, s_contact - 0.005)
        pc, rc, sh = pose(s0, a, b, c, d)
        converge(rob, pc, rc, grip, tol=0.0005, rtol=0.003, n=15, max_pos=0.2)
        s = s0
        jam = 0
        while not rob.ended and rob.step_id - start_step < max_total_steps:
            s_cmd = min(s + step_len, s_goal)
            pc, rc, sh = pose(s_cmd, a, b, c, d)
            rob.step([servo_row(rob.state, pc, rc, grip, max_pos=0.3)])
            s_meas = progress(sh)
            dz = rob.state[2] - pc[2]
            rerr = np.degrees(np.linalg.norm(rot_err(rc, rob.state[3:6])))
            best_s = max(best_s, s_meas)
            if s_meas >= s_cmd - 0.0012:
                s = s_cmd
                jam = 0
            else:
                jam += 1
            if s >= s_goal - 1e-4 or rob.complete:
                log("insert v4: reached goal on attempt %d step %d" % (attempt, rob.step_id))
                break
            if jam >= 3 or abs(dz) > 0.0015 or rerr > 0.8:
                s_contact = s_meas if s_contact is None else min(s_contact, s_meas)
                log("  jam attempt %d off%s at s %.4f/%.4f dz %.4f rerr %.2f step %d" % (
                    attempt, (a, b, c, d), s_meas, total, dz, rerr, rob.step_id))
                break
        if rob.ended or s >= s_goal - 1e-4 or rob.complete:
            break
        sb = max(0.0, (s_contact or 0.0) - 0.005)
        for _ in range(4):
            pc, rc, sh = pose(sb, a, b, c, d)
            rob.step([servo_row(rob.state, pc, rc, grip, max_pos=0.15)])
        attempt += 1
    if not rob.ended and not rob.complete and best_s < s_goal - 0.002:
        log("insert v4: fallback firm push from step %d" % rob.step_id)
        s0 = max(0.0, (s_contact or 0.0) - 0.005)
        pc, rc, sh = pose(s0)
        converge(rob, pc, rc, grip, tol=0.0006, rtol=0.003, n=10, max_pos=0.2)
        for _ in range(80):
            if rob.ended or rob.complete or (rob.remaining is not None and rob.remaining <= 6):
                break
            s_meas = progress(sh)
            pc, rc, sh = pose(min(s_meas + 0.006, s_goal))
            rob.step([servo_row(rob.state, pc, rc, grip, max_pos=0.3)])
            best_s = max(best_s, s_meas)
    log("insert v4 end: step %d best_s %.4f/%.4f attempts %d complete %s ended %s" % (
        rob.step_id, best_s, total, attempt, rob.complete, rob.ended))
    return P, Rv, total


def insert_v5(rob, P, Rv, grip, corr, t0, step_len=0.0012):
    """Aligned approach along the (corrected) video path, then a force-escalating push through the
    resistive hole entry: the target leads the measured progress by L along the path, L grows while
    progress stalls, lateral/orientation tracking stays stiff."""
    keep = [0]
    for j in range(1, len(P)):
        if np.linalg.norm(P[j] - P[keep[-1]]) > 0.001:
            keep.append(j)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    P = P[keep]; Rv = Rv[keep]
    total = sum(np.linalg.norm(P[j + 1] - P[j]) for j in range(len(P) - 1))
    dfin = P[-1] - P[max(0, len(P) - 6)]
    dfin = dfin / max(np.linalg.norm(dfin), 1e-9)
    log("insert v5: path pts %d length %.3f" % (len(P), total))

    def pose(s):
        tp, tr = point_at(P, Rv, s)
        pc, rc = apply_corr(tp, tr, corr)
        return pc, rc, pc - tp

    def progress(shift):
        q = rob.state[:3] - shift
        s_meas, j, dist = path_param(P, q)
        if s_meas > total:
            s_meas = total + np.dot(q - P[-1], dfin)
        return s_meas, dist

    # A: gentle aligned approach until first resistance
    pc, rc, sh = pose(0.0)
    converge(rob, pc, rc, grip, tol=0.0005, rtol=0.003, n=15, max_pos=0.2)
    s = 0.0
    jam = 0
    while not rob.ended and not rob.complete and s < total:
        s_cmd = min(s + step_len, total)
        pc, rc, sh = pose(s_cmd)
        rob.step([servo_row(rob.state, pc, rc, grip, max_pos=0.3)])
        s_meas, _ = progress(sh)
        if s_meas >= s_cmd - 0.0012:
            s = s_cmd; jam = 0
        else:
            jam += 1
            if jam >= 3:
                break
    s_meas, _ = progress(sh)
    log("insert v5: contact at s %.4f/%.4f step %d" % (s_meas, total, rob.step_id))
    # B: force-escalating push
    L = 0.004
    L_max = 0.05
    hist = []
    best_s = s_meas
    n = 0
    while not rob.ended and not rob.complete:
        if rob.remaining is not None and rob.remaining <= 12:
            break
        s_meas, dist = progress(sh)
        best_s = max(best_s, s_meas)
        if s_meas >= total - 0.001:
            log("insert v5: reached path end s %.4f step %d" % (s_meas, rob.step_id))
            break
        hist.append(s_meas)
        if len(hist) >= 5 and n % 5 == 0:
            gain = hist[-1] - hist[-5]
            if gain < 0.0008:
                L = min(L * 1.5, L_max)
            elif gain > 0.003 and L > 0.004:
                L = max(L / 1.5, 0.004)
        L_eff = min(L, max(0.006, 2.0 * (total - s_meas)))
        pc, rc, sh = pose(min(s_meas + L_eff, total + 0.002))
        rerr = np.degrees(np.linalg.norm(rot_err(rc, rob.state[3:6])))
        if rerr > 3.0 or dist > 0.004:
            L = max(0.004, L / 2)
            pc, rc, sh = pose(min(s_meas + L, total + 0.002))
        rob.step([servo_row(rob.state, pc, rc, grip, max_pos=1.0)])
        if n % 5 == 0:
            log("  push n %d step %d s %.4f L %.3f dist %.4f rerr %.2f z %.4f y %.4f" % (
                n, rob.step_id, s_meas, L, dist, rerr, rob.state[2], rob.state[1]))
        n += 1
    # C: hold at the reached pose (no release)
    s_meas, _ = progress(sh)
    hold_s = min(max(s_meas, 0.0), total) if s_meas < total - 0.001 else total + 0.0015
    for _ in range(400):
        if rob.ended or rob.complete or (rob.remaining is not None and rob.remaining <= 6):
            break
        pc, rc, sh = pose(hold_s)
        rob.step([servo_row(rob.state, pc, rc, grip, max_pos=0.2)])
    log("insert v5 end: step %d best_s %.4f/%.4f complete %s ended %s" % (
        rob.step_id, best_s, total, rob.complete, rob.ended))
    return P, Rv, total


def run(rob, t0):
    obs = rob.observe()
    log("instruction:", obs.get("instruction"), "step", rob.step_id, "remaining", rob.remaining)
    vid, exe = load_video(obs)
    if not vid:
        log("no video frames; abort")
        return
    log("video frames", len(vid))
    S = np.array([f["state"] for f in vid], float)
    try:
        if exe is not None:
            d0 = depth_of(vid[0], "front"); de = depth_of(exe, "front")
            diff = np.abs(d0 - de)
            log("layout depth diff: max %.3f m, px>5mm %d" % (diff.max(), int((diff > 0.005).sum())))
    except Exception as e:
        log("layout check failed", e)

    width = S[:, 6] + S[:, 7]
    closed = width < width[0] - 0.004
    close_idx = int(np.argmax(closed)) if closed.any() else None
    log("close idx", close_idx, "width at end", width[-1])
    grip = np.full(len(S), -1.0)
    if close_idx is not None:
        grip[max(0, close_idx - 1):] = 1.0
    N = len(S)
    k_ins = find_insert_start(S, close_idx)
    # pre-contact point: first index after k_ins within 5 cm (horizontal) of the final pose
    dfinal = np.linalg.norm(S[:, :2] - S[-1, :2], axis=1)
    cand = [k for k in range(k_ins, N) if dfinal[k] < 0.05]
    k_pre = cand[0] if cand else k_ins
    # in-air reference frame for the peg measurement: highest z between grasp and k_ins
    if close_idx is not None:
        seg = range(close_idx + 10, k_ins)
        k_air = max(seg, key=lambda k: S[k, 2]) if len(seg) else k_ins
    else:
        k_air = None
    log("k_ins", k_ins, "k_pre", k_pre, "k_air", k_air, "pre pos", np.round(S[k_pre, :3], 4).tolist(),
        "final", np.round(S[-1, :3], 4).tolist())

    # phase 1: temporal replay up to k_pre, with precise convergence before closing and the peg check
    i = 1
    corr = (np.eye(3), np.zeros(3))
    while i <= k_pre and not rob.ended:
        tpos, trot = S[i, :3], S[i, 3:6]
        if close_idx is not None and i == close_idx - 1:
            converge(rob, tpos, trot, -1.0, tol=0.0006, rtol=0.003, n=30, max_pos=0.2)
            log("pre-grasp err %.4f state %s" % (np.linalg.norm(rob.state[:3] - tpos), np.round(rob.state, 4).tolist()))
        if k_air is not None and i == k_air:
            corr = measure_peg_correction(rob, vid, S, k_air, grip[i])
        if i >= k_ins:
            # slow, precise approach to the pre-contact point
            pc, rc = apply_corr(tpos, trot, corr)
            rob.step([servo_row(rob.state, pc, rc, grip[i], max_pos=0.3)])
        else:
            rob.step([servo_row(rob.state, tpos, trot, grip[i])])
        i += 1
        if i % 50 == 0:
            log("i", i, "step", rob.step_id, "err %.4f" % np.linalg.norm(rob.state[:3] - S[min(i, N - 1), :3]),
                "width %.4f" % (rob.state[6] + rob.state[7]), "t %.0f" % (time.time() - t0))
    log("phase1 done step", rob.step_id, "state", np.round(rob.state, 4).tolist(), "w %.4f" % (rob.state[6] + rob.state[7]))
    if rob.state[6] + rob.state[7] < 0.004:
        log("WARNING: gripper fully closed - peg likely not held")
    # phase 2: careful insertion with search
    P, Rv, total = insert_v5(rob, S[k_pre:, :3], S[k_pre:, 3:6], grip[-1], corr, t0)
    # hold
    for _ in range(5):
        if rob.ended:
            break
        rob.step([[0, 0, 0, 0, 0, 0, grip[-1]]])
    log("final step", rob.step_id, "state", np.round(rob.state, 4).tolist(), "complete", rob.complete,
        "ended", rob.ended, "t %.0f" % (time.time() - t0))
