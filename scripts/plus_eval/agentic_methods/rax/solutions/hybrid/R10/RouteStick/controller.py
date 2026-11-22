"""RouteStick controller v1.

Strategy: the conditioning video shows the robot drawing the reference path in the
same scene. Each conditioning frame publishes the robot's measured TCP state, so the
drawn path is recovered from that proprioception and reproduced by closed-loop
waypoint tracking (chunked rows; every chunk re-anchors on the measured TCP).
"""
import json, os, time, traceback
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perception import load_rgbd, find_row_targets, stick_tip, find_bars, world_map, _components
import yawplan
from scipy.spatial.transform import Rotation as Rot
from warp import warp_path

VERSION = "v9"
WS = Path("/workspace")
OUT = WS / "output"
POS_SCALE = 0.05          # metres per unit action
ROT_SCALE = 0.5
CHUNK = 5                 # rows per robomme_step call
GAIN = 0.4                # measured: one row moves ~0.4 of its commanded delta (ep1 probe)
EFF = POS_SCALE * GAIN    # metres actually moved per unit row
MAX_ROW = 0.95            # clip
SETTLE_TOL = 0.004        # m; final-waypoint tolerance

_log_f = None
def log(*a):
    global _log_f
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        if _log_f is None:
            OUT.mkdir(parents=True, exist_ok=True)
            _log_f = open(OUT / "controller_log.txt", "a")
        _log_f.write(s + "\n"); _log_f.flush()
    except Exception:
        pass

def packet(reply):
    if isinstance(reply, dict):
        reply = reply.get("content", [])
    for b in reply or []:
        if isinstance(b, dict) and b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                continue
    return {}

def ended(p):
    if p.get("episode_ended") or (p.get("transition") or {}).get("episode_ended"):
        return True
    for s in (p.get("steps") or []) + ((p.get("transition") or {}).get("steps") or []):
        if s.get("episode_ended"):
            return True
    return False

def complete(p):
    if p.get("task_complete") or (p.get("transition") or {}).get("task_complete"):
        return True
    for s in (p.get("steps") or []) + ((p.get("transition") or {}).get("steps") or []):
        if s.get("task_complete"):
            return True
    return False

def remaining(p):
    return (p.get("episode_limits") or {}).get("remaining_episode_steps")

def load_reference(p, manifest_override=None):
    """Return (src_idx, xyz, rotvec) of conditioning-video TCP states."""
    seq = p.get("frame_sequence") or {}
    mp = seq.get("manifest_path") or manifest_override
    if not mp:
        return None
    m = json.loads((WS / mp).read_text())
    fr = [f for f in m["frames"] if f.get("phase") == "video_demo" and f.get("state")]
    fr.sort(key=lambda f: f.get("frame_index", 0))
    if len(fr) < 5:
        return None
    t = np.array([f.get("source_frame_index", i) for i, f in enumerate(fr)], float)
    S = np.array([f["state"][:6] for f in fr], float)
    ex = [f for f in m["frames"] if f.get("phase") == "execution"]
    return t, S[:, :3], S[:, 3:6], fr[0], (ex[-1] if ex else None)

def frame_targets(f):
    files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
    rgb, dep = load_rgbd(WS / files[("front", "rgb")], WS / files[("front", "depth")],
                         (f.get("depth", {}).get("front", {}) or {}).get("meters_per_unit", 0.001))
    T, cands = find_row_targets(rgb, dep, f["camera_intrinsics"]["front"]["matrix"],
                                f["camera_poses"]["front"])
    return T, cands

DZ_MIN, DZ_MAX = -0.04, 0.10
SHORT_STICK_DZ = 0.02     # above this the hand would sink below the bar tops
HAND_BELOW_TCP = 0.088
BODY_BELOW_TCP = 0.108    # thick hand body starts ~TCP+0.11 (front depth slices); below it the 4x20 cm plate
FINGER_BELOW_TCP = 0.083  # plate bottom ~TCP+0.085
BODY_CLEAR = 0.003
LIFT_CLEAR = 0.012
BODY_BOTTOM_TCP = 0.110   # thick body bottom above TCP (ep9: contact at TCP 0.023 with bar tops 0.131)
BODY_RAISE_CLEAR = 0.006
TARGET_EXEMPT_R = 0.015   # within this of a target keep the ep9-tested low height (activation)
TIP_MAX_ABOVE_VIDEO = 0.003   # rehearsal 2: tip 0.014 above video height drew nothing, no target activated    # lowest visible hand point above TCP (front depth, rehearsal 1)
BAR_CLEAR = 0.012         # hand clearance above bar tops
DIP_R0, DIP_R1 = 0.010, 0.025   # full dip within R0 of a target, fully raised beyond R1
GAIN_R = 0.4              # assumed rotation response (closed loop per chunk)   # tested dz 0.0 (ep1), 0.058 (ep2), 0.048 (ep3)

def tip_offset(f):
    files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
    tcp = np.array(f["state"][:3], float)
    tip, info = stick_tip((WS / files[("wrist", "rgb")], WS / files[("wrist", "depth")]),
                          f["camera_intrinsics"]["wrist"]["matrix"], f["camera_poses"]["wrist"], tcp)
    if tip is None:
        return None, info
    return tip - tcp, info

def chain_ok(A, B):
    la = np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1)
    lb = np.linalg.norm(np.diff(B[:, :2], axis=0), axis=1)
    r = lb / np.maximum(la, 1e-6)
    return bool(np.all(r > 0.6) and np.all(r < 1.6) and la.min() > 0.05), r

def resample(t, P, step_frames=1.0):
    """Linear interpolation to every source frame (one control step each)."""
    tt = np.arange(t[0], t[-1] + 1e-9, step_frames)
    return np.stack([np.interp(tt, t, P[:, k]) for k in range(3)], 1)

def bar_top(f):
    files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
    rgb, dep = load_rgbd(WS / files[("front", "rgb")], WS / files[("front", "depth")])
    P = world_map(dep, f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    z = P[..., 2]
    m = np.isfinite(z) & (P[..., 0] > -0.45) & (z > 0.03) & (z < 0.3)
    # bars: non-gray (colored) or dark pixels; exclude robot (bright gray/white)
    rgbm = rgb[m]; mx = rgbm.max(1); mn = rgbm.min(1)
    bar = ~((mn > 120) & ((mx - mn) < 30))
    zz = z[m][bar]
    return float(np.percentile(zz, 99.5)) if len(zz) > 50 else 0.131

def bar_obstacles(f):
    """xy points of the bar top faces (front depth, colored/dark non-robot pixels)."""
    files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
    rgb, dep = load_rgbd(WS / files[("front", "rgb")], WS / files[("front", "depth")])
    P = world_map(dep, f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    z = P[..., 2]
    mask = np.isfinite(z) & (z > 0.03) & (z < 0.2) & (P[..., 0] > -0.45)
    obs = []
    for c in sorted(_components(mask), key=lambda c: -len(c))[:4]:
        Q = P[c[:, 0], c[:, 1]]
        top = Q[Q[:, 2] > Q[:, 2].max() - 0.004]
        obs.append(top[::max(1, len(top) // 80), :2])
    return np.concatenate(obs) if obs else None

def dip_profile(P, T, z_low, z_high):
    """Per-waypoint TCP z: z_low near targets, z_high elsewhere (only for low path parts)."""
    out = P.copy()
    d = np.min(np.linalg.norm(P[:, None, :2] - T[None, :, :2], axis=2), axis=1)
    s = np.clip((d - DIP_R0) / (DIP_R1 - DIP_R0), 0, 1)
    s = 0.5 - 0.5 * np.cos(np.pi * s)
    zprof = z_low + (z_high - z_low) * s
    out[:, 2] = np.maximum(P[:, 2], zprof)   # never lower than the reference (descent stays)
    return out

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx; self.p = None; self.done = False; self.success = False
    def observe(self):
        self.p = packet(self.ctx.call("robomme_observe")); return self.p
    def pos(self):
        return np.array(self.p["state"][:3], float)
    def step(self, rows):
        rem = remaining(self.p)
        if rem is not None:
            rows = rows[:max(0, rem)]
        if not rows or self.done:
            self.done = True; return self.p
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        p = packet(self.ctx.call("robomme_step", actions=rows))
        if p.get("state") is None:
            p["state"] = self.p["state"]
        self.p = p
        if complete(p): self.success = True
        if ended(p): self.done = True
        return p

def rot_rows(robot, R_des, k):
    if R_des is None:
        return [0.0, 0.0, 0.0]
    R_cur = Rot.from_rotvec(np.array(robot.p["state"][3:6], float))
    e = (R_des * R_cur.inv()).as_rotvec()
    r = e / (ROT_SCALE * GAIN_R) / max(1, k)
    return list(np.clip(r, -0.6, 0.6))

def rotate_to(robot, R_des, n=40, tol=0.03):
    for _ in range(n):
        if robot.done: return
        R_cur = Rot.from_rotvec(np.array(robot.p["state"][3:6], float))
        if np.linalg.norm((R_des * R_cur.inv()).as_rotvec()) < tol: return
        r = rot_rows(robot, R_des, 3)
        robot.step([[0, 0, 0] + r + [-1.0]] * 3)

def follow(robot, W, chunk=CHUNK, R_des=None, Y=None, R0=None):
    """Track waypoint array W (N,3). Each call: first row re-anchors on measured pos.
    Y (N,) optional yaw (rad, world z) relative to base orientation R0."""
    i = 0; N = len(W); maxerr = 0.0
    while i < N and not robot.done:
        cur = robot.pos()
        k = min(chunk, N - i)
        rows = []; prev = cur
        if Y is not None:
            R_des = Rot.from_rotvec([0, 0, float(Y[i + k - 1])]) * R0
        rr = rot_rows(robot, R_des, k)
        for j in range(k):
            d = (W[i + j] - prev) / EFF
            d = np.clip(d, -MAX_ROW, MAX_ROW)
            rows.append([d[0], d[1], d[2]] + rr + [-1.0])
            prev = prev + d * EFF
        robot.step(rows)
        i += k
        err = float(np.linalg.norm(robot.pos() - W[i - 1]))
        maxerr = max(maxerr, err)
        if (i // chunk) % 10 == 0:
            log(f"[follow] i={i}/{N} err={err:.4f} rem={remaining(robot.p)} pos={np.round(robot.pos(),3).tolist()}")
    return maxerr

def settle(robot, target, n=40):
    for _ in range(n):
        if robot.done: return
        e = target - robot.pos()
        if np.linalg.norm(e) < SETTLE_TOL: return
        d = np.clip(e / EFF, -MAX_ROW, MAX_ROW)
        robot.step([[d[0], d[1], d[2], 0, 0, 0, -1.0]])

def main(ctx, manifest_override=None):
    t0 = time.time()
    log(f"RouteStick controller {VERSION} start")
    robot = Robot(ctx)
    try:
        p = robot.observe()
        ref = load_reference(p, manifest_override)
        if ref is None:
            log("no reference trajectory in conditioning video; holding")
            return
        t, P, Rv, f_vid, f_exe = ref
        B = None; dz_c = 0.0
        try:
            A, ca = frame_targets(f_vid)
            B, cb = frame_targets(f_exe) if f_exe is not None else (None, [])
            log(f"video targets {None if A is None else np.round(A,3).tolist()}")
            log(f"exec  targets {None if B is None else np.round(B,3).tolist()}")
            if A is not None and B is not None:
                ok, r = chain_ok(A, B)
                log(f"chain ratio {np.round(r,3).tolist()} ok={ok}")
                if ok:
                    P = warp_path(P, A, B)
                    log("path warped to execution layout")
            else:
                log(f"target detection failed: {len(ca)} / {len(cb)} candidates; raw replay")
        except Exception:
            log("warp failed, raw replay\n" + traceback.format_exc())
        # stick length compensation: keep the physical tip at the video's tip height
        try:
            ov, iv = tip_offset(f_vid)
            oe, ie = tip_offset(f_exe) if f_exe is not None else (None, None)
            log(f"tip offset video {None if ov is None else np.round(ov,4).tolist()} {iv}")
            log(f"tip offset exec  {None if oe is None else np.round(oe,4).tolist()} {ie}")
            if ov is not None and oe is not None:
                dz = float(oe[2] - ov[2])
                dz_c = float(np.clip(dz, DZ_MIN, DZ_MAX))
                log(f"stick dz={dz:.4f} applied={dz_c:.4f}")
                P = P.copy(); P[:, 2] -= dz_c
        except Exception:
            dz_c = 0.0
            log("tip offset failed\n" + traceback.format_exc())
        W = resample(t, P); Y = None; R0 = None; W_base = W.copy()
        try:
            if dz_c > SHORT_STICK_DZ and B is not None and f_exe is not None:
                btop = bar_top(f_exe)
                obs = bar_obstacles(f_exe)
                R0 = Rot.from_rotvec(np.array(robot.p["state"][3:6], float))
                z_vid = float(np.min(W[:, 2]))                      # TCP z giving the video tip height
                z_floor = min(btop + BODY_CLEAR - BODY_BELOW_TCP,    # hand body above bar tops ...
                              z_vid + TIP_MAX_ABOVE_VIDEO)           # ... unless the tip would be too high to draw
                W = W.copy(); W[:, 2] = np.maximum(W[:, 2], z_floor)
                low = W[:, 2] + FINGER_BELOW_TCP < btop + 0.01
                dT = np.min(np.linalg.norm(W[:, None, :2] - B[None, :, :2], axis=2), 1)
                lift_ok = [0] + [i for i in range(1, len(W) - 1)
                                 if dT[i] < 0.012 and dT[i] <= dT[i - 1] and dT[i] <= dT[i + 1]]
                z_lift = btop + LIFT_CLEAR - FINGER_BELOW_TCP
                z_body = btop + BODY_RAISE_CLEAR - BODY_BOTTOM_TCP
                W2, Y2, info = yawplan.plan(W, obs, low, lift_ok, z_lift, z_body=z_body, exempt=dT < TARGET_EXEMPT_R)
                info["z_body"] = round(z_body, 4)
                log(f"short stick v7: bar_top={btop:.3f} z_floor={z_floor:.3f} z_lift={z_lift:.3f} obs={len(obs)} plan={info}")
                if len(W2) < (remaining(robot.p) or 10**6) - 40:
                    W, Y = W2, Y2
                else:
                    log("yaw plan too long for the step budget; plain replay")
        except Exception:
            Y = None; W = W_base
            log("yaw plan failed\n" + traceback.format_exc())
        log(f"reference: {len(t)} frames, src {t[0]:.0f}-{t[-1]:.0f}, zmin={W[:,2].min():.3f} steps={len(W)}")
        settle(robot, W[0], n=10)
        maxerr = follow(robot, W[1:], Y=None if Y is None else Y[1:], R0=R0)
        log(f"follow done maxerr={maxerr:.4f} rem={remaining(robot.p)} t={time.time()-t0:.1f}s")
        settle(robot, W[-1])
        # hold briefly so terminal checks register
        for _ in range(3):
            if robot.done: break
            robot.step([[0, 0, 0, 0, 0, 0, -1.0]] * 5)
        log(f"end: done={robot.done} success={robot.success} pos={robot.pos().round(3).tolist()} rem={remaining(robot.p)} t={time.time()-t0:.1f}s")
    except Exception:
        log("EXCEPTION\n" + traceback.format_exc())
