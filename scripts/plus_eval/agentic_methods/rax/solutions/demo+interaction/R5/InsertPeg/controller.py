"""InsertPeg controller v5: track the end-effector trajectory shown in this episode's
own conditioning video.

The reset sequence (robomme_observe at step 0) carries per-frame robot state for the
conditioning video. The execution scene starts in the same layout as the video's first
frame, so reproducing the video's EE pose trajectory and gripper timing grasps the same
peg at the same end and inserts it into the same side of the box.
"""
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np

WS = Path("/workspace")
OUT = WS / "output"
POS_SCALE = 0.05
ROT_SCALE = 0.5
LOG = []


import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import holefind as HF  # noqa: E402


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------- rotation math
def rv2mat(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat2rv(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / math.sqrt(max(M[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rot_err(rv_target, rv_cur):
    """World-frame rotation vector taking current to target."""
    return mat2rv(rv2mat(rv_target) @ rv2mat(rv_cur).T)


def slerp_rv(rv0, rv1, a):
    R0 = rv2mat(rv0)
    d = mat2rv(rv2mat(rv1) @ R0.T)
    return mat2rv(rv2mat(d * a) @ R0)


# ---------------------------------------------------------------- MCP helpers
def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def find_manifest(pk):
    fs = pk.get("frame_sequence") or {}
    mp = fs.get("manifest_path")
    if mp and (WS / mp).exists():
        return WS / mp
    # fallback: newest reset sequence on disk for this episode
    ep = pk.get("episode_id")
    cands = sorted(glob.glob(str(WS / "runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json")),
                   key=os.path.getmtime, reverse=True)
    for c in cands:
        try:
            m = json.loads(Path(c).read_text())
            if ep is None or m.get("transition", {}).get("episode_id") == ep:
                return Path(c)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------- trajectory
def load_video_traj(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    frames = [f for f in m["frames"] if f.get("phase") == "video_demo" and f.get("state")]
    frames.sort(key=lambda f: f["frame_index"])
    ts = np.array([float(f.get("timestamp_s", i * 0.05)) for i, f in enumerate(frames)])
    S = np.array([f["state"] for f in frames], float)
    return ts, S


def resample(ts, S, dt=0.05):
    """Resample the video states to the control clock (handles downsampled videos)."""
    if len(ts) < 2:
        return S
    t_new = np.arange(ts[0], ts[-1] + 1e-6, dt)
    out = []
    for t in t_new:
        j = int(np.searchsorted(ts, t, side="right") - 1)
        j = min(max(j, 0), len(ts) - 2)
        a = 0.0 if ts[j + 1] == ts[j] else (t - ts[j]) / (ts[j + 1] - ts[j])
        a = min(max(a, 0.0), 1.0)
        p = S[j, :3] * (1 - a) + S[j + 1, :3] * a
        r = slerp_rv(S[j, 3:6], S[j + 1, 3:6], a)
        g = S[j, 6:8] * (1 - a) + S[j + 1, 6:8] * a
        out.append(np.concatenate([p, r, g]))
    return np.array(out)


def gripper_cmds(S, look=4):
    """+1 (close) when the recorded opening is shrinking or already small, -1 otherwise."""
    w = S[:, 6] + S[:, 7]
    n = len(w)
    cmd = np.full(n, -1.0)
    state = -1.0
    for i in range(n):
        fut = w[min(i + look, n - 1)]
        if fut < w[i] - 0.002:
            state = 1.0
        elif fut > w[i] + 0.002:
            state = -1.0
        elif w[i] > 0.077:
            state = -1.0
        cmd[i] = state
    return cmd


# ---------------------------------------------------------------- control
class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.step_id = 0
        self.ended = False
        self.complete = False
        self.remaining = None

    def observe(self):
        pk = packet(self.ctx.call("robomme_observe"))
        self._upd(pk)
        return pk

    def _upd(self, pk):
        if pk.get("state"):
            self.state = np.array(pk["state"], float)
        self.step_id = pk.get("step_id", self.step_id)
        if pk.get("episode_ended"):
            self.ended = True
        tr = pk.get("transition") or {}
        for s in tr.get("steps", []):
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if tr.get("episode_ended"):
            self.ended = True
        lim = pk.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]

    def step(self, rows):
        pk = packet(self.ctx.call("robomme_step", actions=[[float(x) for x in r] for r in rows]))
        self._upd(pk)
        return pk

    def row_to(self, target, grip, pgain=1.0, rgain=1.0):
        cur = self.state
        dp = (target[:3] - cur[:3]) * pgain / POS_SCALE
        dr = rot_err(target[3:6], cur[3:6]) * rgain / ROT_SCALE
        row = np.concatenate([np.clip(dp, -1, 1), np.clip(dr, -1, 1), [grip]])
        return row


def track(robot, T, G, start=0, stop=None, lead=1):
    stop = len(T) if stop is None else min(stop, len(T))
    errs = []
    for i in range(start, stop):
        if robot.ended:
            break
        if robot.remaining is not None and robot.remaining <= 1:
            break
        tgt = T[min(i + lead, len(T) - 1)]
        row = robot.row_to(tgt, G[min(i + lead, len(T) - 1)])
        robot.step([row])
        e = np.linalg.norm(robot.state[:3] - T[min(i + lead, len(T) - 1)][:3])
        errs.append(e)
        if i % 20 == 0:
            log(f"i={i} step={robot.step_id} pos_err={e:.4f} grip={G[i]:+.0f} width={robot.state[6]+robot.state[7]:.4f}"
                f" ee={np.round(robot.state[:3],3).tolist()} tgt={np.round(T[i][:3],3).tolist()}")
    return errs


def settle(robot, target, grip, n=20, tol=0.002):
    for _ in range(n):
        if robot.ended:
            return
        robot.step([robot.row_to(target, grip)])
        if np.linalg.norm(robot.state[:3] - target[:3]) < tol and np.linalg.norm(rot_err(target[3:6], robot.state[3:6])) < 0.01:
            return



# ---------------------------------------------------------------- perception
import base64, io


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(depth_m, K, pose, mask):
    v, u = np.nonzero(mask)
    d = depth_m[v, u]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pc = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], 1)
    R = quat_mat(pose["quaternion_wxyz"])
    return pc @ R.T + np.array(pose["position_m"])


def peg_points_ee(depth_mm, K, pose, ee_state, max_d=0.11):
    """Near-field (held peg) points of the wrist depth image, in the EE frame."""
    d = depth_mm.astype(float) * 0.001
    mask = (d > 0.02) & (d < max_d)
    mask[196:, :] = False  # finger region
    if mask.sum() < 300:
        return None
    pw = backproject(d, K, pose, mask)
    Ree = rv2mat(ee_state[3:6])
    return (pw - ee_state[:3]) @ Ree  # R^T (p - t)


def load_png_array(path_or_bytes):
    from PIL import Image
    if isinstance(path_or_bytes, (bytes, bytearray)):
        im = Image.open(io.BytesIO(path_or_bytes))
    else:
        im = Image.open(path_or_bytes)
    return np.asarray(im)


def reply_depth(reply, pk, camera="wrist"):
    idx = None
    for a in pk.get("attachments", []):
        if a.get("camera") == camera and a.get("kind") == "depth":
            idx = a.get("content_index")
    if idx is None:
        return None
    b = reply["content"][idx]
    return load_png_array(base64.b64decode(b["data"]))


def video_frame_depth(frame, camera="wrist"):
    for f in frame.get("files", []):
        if f.get("camera") == camera and f.get("kind") == "depth":
            p = WS / f["path"]
            if p.exists():
                return load_png_array(str(p))
    return None


# ---------------------------------------------------------------- plan from video
def plan_from_video(T, G):
    n = len(T)
    ci = int(np.argmax(G > 0)) if (G > 0).any() else None
    if ci is None:
        return None
    z_close = T[ci, 2]
    # hold frame: first index after close where EE is high and still
    hold = None
    sp = np.r_[np.linalg.norm(np.diff(T[:, :3], axis=0), axis=1), 0]
    rs = np.r_[[np.linalg.norm(rot_err(T[i + 1, 3:6], T[i, 3:6])) for i in range(n - 1)], 0]
    for i in range(ci, n - 5):
        if T[i, 2] > z_close + 0.08 and sp[i:i + 5].max() < 0.0015 and rs[i:i + 5].max() < 0.01:
            hold = i + 2
            break
    # pre-insertion index: from the end backwards, the last index farther than PRE from final
    PRE = 0.06
    dist = np.linalg.norm(T[:, :3] - T[-1, :3], axis=1)
    k = n - 1
    while k > ci and dist[k] < PRE:
        k -= 1
    k_pre = k + 1 if k + 1 < n else n - 1
    return {"close": ci, "hold": hold, "pre": k_pre}


SEARCH = [(0, 0), (0, 1.5), (0, -1.5), (1.5, 0), (-1.5, 0), (0, 3), (3, 0), (-3, 0), (0, -3),
          (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5), (0, 4.5), (4.5, 0), (-4.5, 0),
          (3, 3), (-3, 3), (3, -3), (-3, -3), (0, -4.5)]


def goto(robot, target, grip, n, tol=0.0015, pg=1.0, rg=1.0):
    for _ in range(n):
        if robot.ended or (robot.remaining is not None and robot.remaining <= 1):
            return False
        robot.step([robot.row_to(target, grip, pg, rg)])
        if np.linalg.norm(robot.state[:3] - target[:3]) < tol and np.linalg.norm(rot_err(target[3:6], robot.state[3:6])) < 0.02:
            return True
    return False


def insert(robot, P0, Pend, rot, grip, corr):
    a = Pend - P0
    L = float(np.linalg.norm(a))
    a = a / max(L, 1e-9)
    u = np.cross([0, 0, 1.0], a)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    log(f"insert L={L:.4f} axis={np.round(a,3).tolist()} corr={np.round(corr*1000,2).tolist()}mm")
    best = 0.0
    for si, (du, dw) in enumerate(SEARCH):
        if robot.ended or (robot.remaining is not None and robot.remaining <= 3):
            break
        off = corr + (du * u + dw * w) * 0.001
        s_start = max(0.0, best - 0.012) if si > 0 else 0.0
        # retreat along axis (no offset), then shift laterally, then push
        if si > 0:
            cur_s = float(np.dot(robot.state[:3] - (P0 + corr), a))
            back = P0 + corr + a * max(0.0, min(cur_s, best) - 0.012)
            goto(robot, np.r_[back, rot], grip, 12, tol=0.0015, pg=1.0, rg=1.0)
            goto(robot, np.r_[P0 + off + a * s_start, rot], grip, 8, tol=0.001, pg=1.0, rg=1.0)
        hist = []
        s_cmd = s_start
        while not robot.ended and (robot.remaining is None or robot.remaining > 2):
            s_meas = float(np.dot(robot.state[:3] - (P0 + off), a))
            hist.append(s_meas)
            if s_meas > best:
                best = s_meas
            if s_meas >= L - 0.0015:
                log(f"reached depth s={s_meas:.4f} at try {si}")
                # keep pushing gently to complete
                for _ in range(25):
                    if robot.ended:
                        break
                    tgt = np.r_[P0 + off + a * (L + 0.004), rot]
                    robot.step([robot.row_to(tgt, grip, 0.6, 0.5)])
                return True
            if len(hist) > 12 and hist[-1] - hist[-12] < 0.0008:
                log(f"stall try {si} off={np.round(off*1000,1).tolist()} s={s_meas:.4f} best={best:.4f} step={robot.step_id}")
                break
            s_cmd = min(s_cmd + 0.0012, s_meas + 0.005, L + 0.004)
            tgt = np.r_[P0 + off + a * s_cmd, rot]
            # along-axis stiff, lateral compliant-ish
            cur = robot.state
            e = tgt[:3] - cur[:3]
            ea = np.dot(e, a) * a
            el = e - ea
            dp = (ea * 1.0 + el * 0.6) / POS_SCALE
            dr = rot_err(rot, cur[3:6]) * 0.5 / ROT_SCALE
            robot.step([np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip]])
    return False


def servo_pi(robot, tgt, grip, max_steps, tol, integ, kp=1.0, ki=0.3, rtol=0.003, imax=0.004):
    """PI servo on the measured pose; the integral removes steady-state sag and builds
    contact force when the target is blocked."""
    for _ in range(max_steps):
        if robot.ended or (robot.remaining is not None and robot.remaining <= 1):
            return False
        cur = robot.state
        e = tgt[:3] - cur[:3]
        er = rot_err(tgt[3:6], cur[3:6])
        if np.linalg.norm(e) < tol and np.linalg.norm(er) < rtol:
            return True
        integ["p"] = np.clip(integ["p"] + ki * e, -imax, imax)
        integ["r"] = np.clip(integ["r"] + ki * er, -0.02, 0.02)
        dp = (kp * e + integ["p"]) / POS_SCALE
        dr = (er + integ["r"]) / ROT_SCALE
        robot.step([np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip]])
    return False


def insertion_waypoints(T, pre, spacing=0.0015):
    wps = [T[pre, :6].copy()]
    for i in range(pre + 1, len(T)):
        if np.linalg.norm(T[i, :3] - wps[-1][:3]) >= spacing:
            wps.append(T[i, :6].copy())
    if np.linalg.norm(T[-1, :3] - wps[-1][:3]) > 1e-4:
        wps.append(T[-1, :6].copy())
    return wps


SEARCH2 = [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.8, 0.8), (-0.8, 0.8), (0.8, -0.8),
           (-0.8, -0.8), (1.2, 0), (-1.2, 0), (0, 1.2), (0, -1.2)]


def insert_follow(robot, T, pre, grip, corr):
    wps = insertion_waypoints(T, pre)
    P0, Pend = T[pre, :3], T[-1, :3]
    a = Pend - P0
    L = float(np.linalg.norm(a))
    a = a / max(L, 1e-9)
    u = np.cross([0, 0, 1.0], a)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    w = np.cross(a, u)
    log(f"insert_follow: {len(wps)} waypoints L={L:.4f} axis={np.round(a,3).tolist()}")
    integ = {"p": np.zeros(3), "r": np.zeros(3)}
    best = -1.0
    for si, (du, dw) in enumerate(SEARCH2):
        off = corr + (du * u + dw * w) * 0.001
        if si > 0:
            # retreat ~10 mm along the path, reset lateral integral
            s_now = float(np.dot(robot.state[:3] - P0, a))
            k_back = max(0, max(j for j in range(len(wps)) if np.dot(wps[j][:3] - P0, a) <= max(0.0, s_now - 0.010)) if any(np.dot(wq[:3] - P0, a) <= max(0.0, s_now - 0.010) for wq in wps) else 0)
            integ = {"p": np.zeros(3), "r": np.zeros(3)}
            tgt = wps[k_back].copy()
            tgt[:3] += off
            servo_pi(robot, tgt, grip, 15, 0.0006, integ)
            start = k_back
        else:
            start = 0
        last_prog_step = robot.step_id
        k = start
        while k < len(wps):
            if robot.ended or (robot.remaining is not None and robot.remaining <= 2):
                return False
            tgt = wps[k].copy()
            tgt[:3] += off
            ok = servo_pi(robot, tgt, grip, 6, 0.0004, integ)
            s_meas = float(np.dot(robot.state[:3] - P0, a))
            if s_meas > best + 0.0005:
                best = s_meas
                last_prog_step = robot.step_id
            if ok or np.dot(tgt[:3] - robot.state[:3], a) < 0.003:
                k += 1  # converged, or axial lag small enough: advance the target
            if robot.step_id - last_prog_step > 30:
                log(f"stall try {si} off=({du},{dw})mm s={s_meas:.4f} best={best:.4f} wp={k}/{len(wps)} step={robot.step_id}")
                break
        else:
            s_meas = float(np.dot(robot.state[:3] - P0, a))
            log(f"path done try {si} s={s_meas:.4f} L={L:.4f} step={robot.step_id}")
            # final push a little beyond the video end
            tgt = wps[-1].copy()
            tgt[:3] += off + a * 0.003
            servo_pi(robot, tgt, grip, 25, 0.0003, integ)
            if robot.ended or s_meas >= L - 0.002:
                return True
    return False


SEARCH3 = [(0, 0), (0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3), (0.6, 0), (-0.6, 0), (0, 0.6), (0, -0.6),
           (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4), (0.9, 0), (-0.9, 0)]


def insert_timed(robot, T, pre, grip):
    """Replay the video's insertion frames in time with a PI servo (lead 1 frame)."""
    n = len(T)
    P0, Pend = T[pre, :3], T[-1, :3]
    a = Pend - P0
    L = float(np.linalg.norm(a))
    a = a / max(L, 1e-9)
    u = np.cross([0, 0, 1.0], a)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    w = np.cross(a, u)
    sv = np.array([np.dot(T[i, :3] - P0, a) for i in range(n)])
    log(f"insert_timed: frames {pre}..{n-1} L={L:.4f} axis={np.round(a,3).tolist()}")
    best = -1.0
    i0 = pre
    for si, (du, dw) in enumerate(SEARCH3):
        off = (du * u + dw * w) * 0.001
        integ = {"p": np.zeros(3), "r": np.zeros(3)}
        if si > 0:
            s_now = float(np.dot(robot.state[:3] - P0, a))
            s_back = max(min(s_now, best) - 0.020, sv[pre] - 0.01)
            cands = [i for i in range(pre - 40 if pre >= 40 else 0, n) if sv[i] <= s_back]
            i0 = cands[-1] if cands else pre
            # retreat along the video path (reverse) to frame i0, then re-servo with the offset
            tgt = T[i0, :6].copy()
            tgt[:3] += off
            servo_pi(robot, tgt, grip, 20, 0.0004, integ)
            integ = {"p": np.zeros(3), "r": np.zeros(3)}
            servo_pi(robot, tgt, grip, 10, 0.0003, integ)
        i = i0
        stuck = 0
        while i < n:
            if robot.ended or (robot.remaining is not None and robot.remaining <= 2):
                return False
            j = min(i + 1, n - 1)
            tgt = T[j, :6].copy()
            tgt[:3] += off
            cur = robot.state
            e = tgt[:3] - cur[:3]
            er = rot_err(tgt[3:6], cur[3:6])
            integ["p"] = np.clip(integ["p"] + 0.3 * e, -0.004, 0.004)
            integ["r"] = np.clip(integ["r"] + 0.3 * er, -0.02, 0.02)
            dp = (e + integ["p"]) / POS_SCALE
            dr = (er + integ["r"]) / ROT_SCALE
            robot.step([np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip]])
            s_meas = float(np.dot(robot.state[:3] - P0, a))
            if s_meas > best + 0.0003:
                best = s_meas
                stuck = 0
            else:
                stuck += 1
            lag = sv[j] - s_meas
            if lag < 0.003:
                i += 1  # advance in time only while we keep up with the video
            if stuck > 30 and lag > 0.002:
                log(f"stall try {si} off=({du},{dw})mm s={s_meas:.4f} best={best:.4f} frame={i} step={robot.step_id}"
                    f" lat_u={1000*np.dot(robot.state[:3]-T[j,:3],u):.2f} lat_w={1000*np.dot(robot.state[:3]-T[j,:3],w):.2f}")
                break
        else:
            s_meas = float(np.dot(robot.state[:3] - P0, a))
            log(f"video path done try {si} s={s_meas:.4f} L={L:.4f} step={robot.step_id}")
            tgt = T[-1, :6].copy()
            tgt[:3] += off + a * 0.003
            servo_pi(robot, tgt, grip, 30, 0.0002, integ)
            return True
    return False


def _axes(T, pre):
    P0, Pend = T[pre, :3], T[-1, :3]
    a = Pend - P0
    L = float(np.linalg.norm(a))
    a = a / max(L, 1e-9)
    u = np.cross([0, 0, 1.0], a)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    w = np.cross(a, u)
    return P0, a, u, w, L


def replay_from(robot, T, i0, grip, off, P0, a, sv, integ, stall_n=20):
    """Timed replay of video frames i0.. with lateral offset `off`.
    Returns ("done", s) or ("stall", s, frame)."""
    n = len(T)
    best = -1.0
    stuck = 0
    i = i0
    while i < n:
        if robot.ended or (robot.remaining is not None and robot.remaining <= 2):
            return ("end", float(np.dot(robot.state[:3] - P0, a)), i)
        j = min(i + 1, n - 1)
        tgt = T[j, :6].copy()
        tgt[:3] += off
        cur = robot.state
        e = tgt[:3] - cur[:3]
        er = rot_err(tgt[3:6], cur[3:6])
        integ["p"] = np.clip(integ["p"] + 0.3 * e, -0.004, 0.004)
        integ["r"] = np.clip(integ["r"] + 0.3 * er, -0.02, 0.02)
        dp = (e + integ["p"]) / POS_SCALE
        dr = (er + integ["r"]) / ROT_SCALE
        robot.step([np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip]])
        s_meas = float(np.dot(robot.state[:3] - P0, a))
        if s_meas > best + 0.0003:
            best = s_meas
            stuck = 0
        else:
            stuck += 1
        lag = sv[j] - s_meas
        if lag < 0.003:
            i += 1
        if stuck > stall_n and lag > 0.002:
            return ("stall", s_meas, i)
    return ("done", float(np.dot(robot.state[:3] - P0, a)), n - 1)


def spiral_search(robot, T, j, grip, base_off, s_face, P0, a, u, w, rmax=0.003, pitch=0.0004,
                  ds=0.0004, preload=0.0025, enter=0.0015):
    """Compliant Archimedean spiral on the contact face: axial target s_face+preload,
    lateral offsets around base_off.  Returns lateral offset at entry or None."""
    rot = T[j, 3:6]
    # lateral reference = video path point at frame j projected to the face
    ref = T[j, :3] - a * np.dot(T[j, :3] - P0, a)
    integ_l = np.zeros(3)
    integ_r = np.zeros(3)
    th = 0.0
    steps = 0
    max_steps = int(np.pi * rmax * rmax / pitch / ds) + 40
    while steps < max_steps:
        if robot.ended or (robot.remaining is not None and robot.remaining <= 3):
            return None
        r = pitch * th / (2 * np.pi)
        if r > rmax:
            break
        off = base_off + r * (np.cos(th) * u + np.sin(th) * w)
        tgt = ref + a * (s_face + preload) + off
        cur = robot.state
        e = tgt - cur[:3]
        e_ax = np.dot(e, a)
        e_lat = e - e_ax * a
        integ_l = np.clip(integ_l + 0.4 * e_lat, -0.003, 0.003)
        er = rot_err(rot, cur[3:6])
        integ_r = np.clip(integ_r + 0.3 * er, -0.02, 0.02)
        dp = (e_lat + integ_l + min(e_ax, preload) * a) / POS_SCALE
        dr = (er + integ_r) / ROT_SCALE
        robot.step([np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip]])
        steps += 1
        s_meas = float(np.dot(robot.state[:3] - P0, a))
        if steps % 25 == 0:
            act = robot.state[:3] - ref - a * np.dot(robot.state[:3] - ref, a)
            log(f"  spiral k={steps} r={r*1000:.2f} s={s_meas:.4f} cmd={np.round(off*1000,2).tolist()} act={np.round(act*1000,2).tolist()}")
        if s_meas > s_face + enter:
            act = robot.state[:3] - ref - a * np.dot(robot.state[:3] - ref, a)
            log(f"spiral ENTRY s={s_meas:.4f} face={s_face:.4f} r={r*1000:.2f}mm off_cmd={np.round(off*1000,2).tolist()}"
                f" off_act={np.round(act*1000,2).tolist()} steps={steps} step={robot.step_id}")
            return robot.state[:3] - ref - a * np.dot(robot.state[:3] - ref, a)
        # arc-length parametrisation: dtheta = ds / max(r, small)
        th += ds / max(r, 0.0006)
    log(f"spiral no entry face={s_face:.4f} rmax={rmax*1000:.1f}mm steps={steps} step={robot.step_id}")
    return None


def video_hole_ref(vframes, P0, a, u, ref):
    """Hole opening seen in the video's wrist depth while the expert approached the face."""
    cands = []
    for vf in vframes:
        st = np.array(vf["state"][:3], float)
        sv_ = float((st - P0) @ a)
        if 0.004 < sv_ < 0.024:
            cands.append((sv_, vf))
    cands.sort(key=lambda c: -c[0])
    hs = []
    for sv_, vf in cands[::2][:10]:
        try:
            d = video_frame_depth(vf)
            if d is None:
                continue
            dm = d.astype(float) * 0.001
            Q = backproject(dm, vf["camera_intrinsics"]["wrist"]["matrix"], vf["camera_poses"]["wrist"], dm > 0)
            h = HF.detect_hole([Q], P0, a, u, ref, log=lambda *x: None)
            if h is not None and h["n"] >= 60:
                hs.append(h)
        except Exception as ex:
            log("video hole frame failed", repr(ex))
    if not hs:
        log("video hole ref: none")
        return None
    out = {k: float(np.median([h[k] for h in hs])) for k in ("u_lo", "u_hi", "z_floor", "face_s")}
    log("video hole ref", len(hs), {k: round(v * 1000, 1) for k, v in out.items()})
    return out


def look_hole(robot, T, j, grip, off, P0, a, u, ref):
    """Back off 3 mm from the face, observe from three lateral positions, detect the hole."""
    clouds = []
    base = T[j, :6].copy()
    base[:3] += off - a * 0.004
    for du in (0.0, 0.015, -0.015):
        if robot.ended or (robot.remaining is not None and robot.remaining < 60):
            break
        tgt = base.copy()
        tgt[:3] += du * u
        servo_pi(robot, tgt, grip, 14, 0.0008, {"p": np.zeros(3), "r": np.zeros(3)})
        try:
            reply = robot.ctx.call("robomme_observe")
            pk = packet(reply)
            robot._upd(pk)
            d = reply_depth(reply, pk)
            dm = d.astype(float) * 0.001
            clouds.append(backproject(dm, pk["camera_intrinsics"]["wrist"]["matrix"], pk["camera_poses"]["wrist"], dm > 0))
        except Exception as ex:
            log("look failed", repr(ex))
    if not clouds:
        return None
    return HF.detect_hole(clouds, P0, a, u, ref, log=log)


def hole_shift(h, hr, peg_u):
    """Lateral (u) and vertical shift of our hole relative to the video's hole."""
    W = hr["u_hi"] - hr["u_lo"]
    c_ref = 0.5 * (hr["u_lo"] + hr["u_hi"])
    width = h["u_hi"] - h["u_lo"]
    if width > W - 0.002:
        c = 0.5 * (h["u_lo"] + h["u_hi"])
    elif abs(h["u_hi"] - peg_u) > abs(h["u_lo"] - peg_u):
        c = h["u_hi"] - 0.5 * W
    else:
        c = h["u_lo"] + 0.5 * W
    return c - c_ref, h["z_floor"] - hr["z_floor"]


def insert_spiral(robot, T, pre, grip, vframes=None):
    """v6: timed replay; on a stall at the face, compliant spiral search; on entry, resume
    the replay with the found lateral offset.  Repeats for later stalls."""
    n = len(T)
    P0, a, u, w, L = _axes(T, pre)
    sv = np.array([np.dot(T[i, :3] - P0, a) for i in range(n)])
    log(f"insert_spiral: frames {pre}..{n-1} L={L:.4f} axis={np.round(a,3).tolist()}")
    off = np.zeros(3)
    i0 = pre
    spirals = 0
    looked = False
    href = None
    if vframes is not None:
        try:
            href = video_hole_ref(vframes, P0, a, u, T[-1, :3])
        except Exception as ex:
            log("video hole ref failed", repr(ex))
    plan = [(0.003, 0.0004, 0.0025), (0.0045, 0.0005, 0.0015), (0.002, 0.00025, 0.004),
            (0.006, 0.0006, 0.002), (0.003, 0.0003, 0.001)]
    while True:
        integ = {"p": np.zeros(3), "r": np.zeros(3)}
        res = replay_from(robot, T, i0, grip, off, P0, a, sv, integ)
        if res[0] == "end":
            return False
        if res[0] == "done":
            log(f"video path done s={res[1]:.4f} L={L:.4f} step={robot.step_id} off={np.round(off*1000,2).tolist()}")
            tgt = T[-1, :6].copy()
            tgt[:3] += off + a * 0.003
            servo_pi(robot, tgt, grip, 30, 0.0002, integ)
            return True
        s_face, j = res[1], res[2]
        log(f"stall s={s_face:.4f} frame={j} step={robot.step_id} off={np.round(off*1000,2).tolist()}")
        if not looked and vframes is not None:
            looked = True
            try:
                if href is None:
                    pass
                else:
                    h = look_hole(robot, T, j, grip, off, P0, a, u, T[-1, :3])
                    if h is not None:
                        du, dz = hole_shift(h, href, float(np.dot(off, u)))
                        log(f"hole shift du={du*1000:.1f}mm dz={dz*1000:.1f}mm")
                        if abs(du) < 0.03 and abs(dz) < 0.02:
                            off = off + du * u + np.array([0.0, 0.0, dz])
                            # withdraw 6 mm, move to the new line, then resume the replay
                            k_back = [k for k in range(pre, n) if sv[k] <= s_face - 0.006]
                            i0 = k_back[-1] if k_back else pre
                            tgt = T[i0, :6].copy()
                            tgt[:3] += off
                            mid = robot.state[:6].copy()
                            mid[:3] = mid[:3] - a * 0.004
                            servo_pi(robot, mid, grip, 10, 0.001, {"p": np.zeros(3), "r": np.zeros(3)})
                            servo_pi(robot, tgt, grip, 25, 0.0005, {"p": np.zeros(3), "r": np.zeros(3)})
                            log(f"moved to hole line off={np.round(off*1000,1).tolist()} step={robot.step_id}")
                            continue
            except Exception as ex:
                log("hole look failed", repr(ex))
        if spirals >= len(plan) or robot.remaining is not None and robot.remaining < 25:
            return False
        rmax, pitch, preload = plan[spirals]
        if spirals >= 1:
            # back off 3 mm from the face before a new spiral (unload, re-seat)
            tgt = T[j, :6].copy()
            tgt[:3] += off - a * 0.003
            servo_pi(robot, tgt, grip, 10, 0.0005, {"p": np.zeros(3), "r": np.zeros(3)})
        spirals += 1
        found = spiral_search(robot, T, j, grip, off, s_face, P0, a, u, w, rmax=rmax, pitch=pitch,
                              preload=preload)
        if found is not None:
            off = found
        s_now = float(np.dot(robot.state[:3] - P0, a))
        cands = [k for k in range(pre, n) if sv[k] <= s_now]
        i0 = cands[-1] if cands else pre


def grasp_at(robot, T, G, ci, tries=3):
    """Converge to the video's grasp pose with the gripper open, then close while holding."""
    w = T[:, 6] + T[:, 7]
    n = len(T)
    j = ci
    while j < n - 1 and w[j] > w[min(ci + 30, n - 1)] + 0.001:
        j += 1
    # pose where the video's fingers finished closing (EE is stationary there)
    gp = T[j].copy()
    above = gp.copy()
    above[2] += 0.04
    held_w = float(w[j])
    for t in range(tries):
        ok = goto(robot, np.r_[gp[:6]], -1.0, 25, tol=0.0012)
        log(f"grasp try {t}: at pose ok={ok} ee={np.round(robot.state[:3],4).tolist()} tgt={np.round(gp[:3],4).tolist()}")
        for _ in range(10):
            if robot.ended:
                return False
            robot.step([robot.row_to(gp, 1.0)])
        width = robot.state[6] + robot.state[7]
        log(f"grasp try {t}: width={width:.4f} video={held_w:.4f}")
        if width > 0.004 and abs(width - held_w) < 0.008:
            return True
        # failed: open, back off upward, retry
        for _ in range(8):
            robot.step([robot.row_to(above, -1.0)])
        goto(robot, above, -1.0, 15, tol=0.002)
    return False


def _main(ctx, stop_at=None, start_idx=0):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    robot = Robot(ctx)
    pk = robot.observe()
    log("start step", robot.step_id, "state", np.round(robot.state, 3).tolist())
    mp = find_manifest(pk)
    log("manifest", mp)
    if mp is None:
        log("ERROR: no reset manifest; nothing to imitate")
        return
    man = json.loads(Path(mp).read_text())
    vframes = sorted([f for f in man["frames"] if f.get("phase") == "video_demo" and f.get("state")],
                     key=lambda f: f["frame_index"])
    ts, S = load_video_traj(mp)
    T = resample(ts, S)
    G = gripper_cmds(T)
    plan = plan_from_video(T, G)
    log("video frames", len(S), "resampled", len(T), "plan", plan)
    np.save(OUT / "traj.npy", T)
    if plan is None:
        track(robot, T, G)
        return robot
    hold, pre = plan["hold"], plan["pre"]

    ci = plan["close"]
    goto(robot, T[start_idx], -1.0, 25 if start_idx else 10, tol=0.003)
    corr = np.zeros(3)
    # approach with gripper open, stop a few frames before the video starts closing
    track(robot, T, np.full(len(G), -1.0), start=start_idx, stop=max(start_idx, ci - 2))
    grasped = grasp_at(robot, T, G, ci)
    log("grasped", grasped, "width", round(float(robot.state[6] + robot.state[7]), 4))
    gstart = ci + 1
    w = T[:, 6] + T[:, 7]
    while gstart < len(T) - 1 and w[gstart] > w[min(ci + 30, len(T) - 1)] + 0.001:
        gstart += 1
    if hold is not None and gstart < hold < pre:
        track(robot, T, G, start=gstart, stop=hold)
        goto(robot, T[hold], G[hold], 15, tol=0.0008)
        try:
            reply = ctx.call("robomme_observe")
            opk = packet(reply)
            robot._upd(opk)
            d_our = reply_depth(reply, opk)
            # nearest original video frame to the hold time
            th = ts[0] + hold * 0.05
            vf = min(vframes, key=lambda f: abs(float(f.get("timestamp_s", 0)) - th))
            d_vid = video_frame_depth(vf)
            P_our = peg_points_ee(d_our, opk["camera_intrinsics"]["wrist"]["matrix"], opk["camera_poses"]["wrist"], robot.state)
            P_vid = peg_points_ee(d_vid, vf["camera_intrinsics"]["wrist"]["matrix"], vf["camera_poses"]["wrist"], np.array(vf["state"]))
            log("peg points", None if P_our is None else len(P_our), None if P_vid is None else len(P_vid))
            if P_our is not None and P_vid is not None:
                e_ee = np.median(P_our, 0) - np.median(P_vid, 0)
                log("in-hand offset (EE frame, mm)", np.round(e_ee * 1000, 2).tolist(), "n", len(P_our), len(P_vid))
                # v5: logged only; offsets are at depth-quantisation level and the grasp
                # replicates the video, so no correction is applied.
        except Exception as ex:
            log("in-hand measurement failed", repr(ex))
        track(robot, T, G, start=hold, stop=pre)
    else:
        track(robot, T, G, start=gstart, stop=pre)
    if stop_at == "pre":
        return robot
    P0 = T[pre, :3]
    rot = T[-1, 3:6]
    gr = G[-1]
    integ0 = {"p": np.zeros(3), "r": np.zeros(3)}
    servo_pi(robot, T[pre, :6], gr, 20, 0.0004, integ0)
    ok = insert_spiral(robot, T, pre, gr, vframes=vframes)
    log("done step", robot.step_id, "ended", robot.ended, "complete", robot.complete, "reached", ok,
        "wall", round(time.time() - t0, 1))
    try:
        (OUT / "controller_log.txt").write_text("\n".join(LOG))
    except Exception:
        pass
    return robot


def main(ctx, stop_at=None, start_idx=0):
    """Entry point: never raises, so a perception/planning bug cannot crash the episode."""
    import traceback
    try:
        return _main(ctx, stop_at=stop_at, start_idx=start_idx)
    except Exception as ex:
        log("CONTROLLER EXCEPTION", repr(ex))
        log(traceback.format_exc())
        try:
            (OUT / "controller_log.txt").write_text("\n".join(LOG))
        except Exception:
            pass
        return None
