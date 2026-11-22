"""InsertPeg controller v1: imitate the current episode's conditioning video.

The conditioning video (supplied at reset, same episode) carries the robot's
proprioceptive state per frame. Execution starts from the same layout as the
video start, so the controller extracts the demonstrated grasp pose (where the
gripper closed) and the post-grasp key poses (lift, re-orient, transport,
insertion stroke) and reproduces them with closed-loop servoing on measured state.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

WS = Path("/workspace")
OUT = WS / "output"
LOG = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


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
        self.remaining = None
        self.bias = np.zeros(3)
        self.observe()

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self._absorb(p)
        return p

    def _absorb(self, p):
        if "state" in p:
            self.state = np.array(p["state"], dtype=float)
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

    def step(self, rows):
        if self.ended:
            return {}
        p = packet(self.ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows]))
        self._absorb(p)
        return p

    @property
    def pos(self):
        return self.state[:3]

    @property
    def rot(self):
        return R.from_rotvec(self.state[3:6])

    @property
    def width(self):
        return float(self.state[6] + self.state[7])

    def action_to(self, pos, rot, grip, vmax=0.6, wmax=0.4):
        dp = (np.asarray(pos) - self.pos) / 0.05
        n = np.linalg.norm(dp)
        if n > vmax:
            dp = dp * vmax / n
        dr = (rot * self.rot.inv()).as_rotvec() / 0.5
        m = np.linalg.norm(dr)
        if m > wmax:
            dr = dr * wmax / m
        return list(np.clip(dp, -1, 1)) + list(np.clip(dr, -1, 1)) + [grip]

    def err(self, pos, rot):
        return (float(np.linalg.norm(np.asarray(pos) - self.pos)),
                float(np.degrees((rot * self.rot.inv()).magnitude())))

    def move_to(self, pos, rot, grip, tol=0.003, tol_deg=2.0, max_steps=60, vmax=0.6, wmax=0.4, chunk=1):
        steps = 0
        while steps < max_steps and not self.ended:
            e, ed = self.err(pos, rot)
            if e < tol and ed < tol_deg:
                break
            self.step([self.action_to(pos, rot, grip, vmax, wmax)] * 1)
            steps += 1
        e, ed = self.err(pos, rot)
        return e, ed, steps

    def servo(self, pos, rot, grip, tol=0.0008, tol_deg=1.0, max_steps=60, vmax=0.3, wmax=0.3, ki=0.25, bmax=0.006):
        """Integral-corrected servo: compensates steady-state tracking error under load."""
        pos = np.asarray(pos, dtype=float)
        steps = 0
        while steps < max_steps and not self.ended:
            e = pos - self.pos
            ed = np.degrees((rot * self.rot.inv()).magnitude())
            if np.linalg.norm(e) < tol and ed < tol_deg:
                break
            self.bias = np.clip(self.bias + ki * e, -bmax, bmax)
            dp = (e + self.bias) / 0.05
            n = np.linalg.norm(dp)
            if n > vmax:
                dp = dp * vmax / n
            dr = (rot * self.rot.inv()).as_rotvec() / 0.5
            m = np.linalg.norm(dr)
            if m > wmax:
                dr = dr * wmax / m
            self.step([list(np.clip(dp, -1, 1)) + list(np.clip(dr, -1, 1)) + [grip]])
            steps += 1
        e, ed = self.err(pos, rot)
        return e, ed, steps

    def hold_bias(self, grip, n, pos, rot):
        """Hold a pose with the integral correction (settles steady-state error before measuring)."""
        for _ in range(n):
            if self.ended:
                break
            e = np.asarray(pos) - self.pos
            self.bias = np.clip(self.bias + 0.25 * e, -0.006, 0.006)
            dp = np.clip((e + self.bias) / 0.05, -0.2, 0.2)
            dr = np.clip((rot * self.rot.inv()).as_rotvec() / 0.5, -0.2, 0.2)
            self.step([list(dp) + list(dr) + [grip]])

    def hold(self, grip, n):
        pos, rot = self.pos.copy(), self.rot
        for _ in range(n):
            if self.ended:
                break
            self.step([self.action_to(pos, rot, grip, 0.3, 0.2)])


def load_video(p0):
    seq = p0.get("frame_sequence") or {}
    mpath = seq.get("manifest_path")
    man = json.loads((WS / mpath).read_text())
    frames = [f for f in man["frames"] if f.get("phase") == "video_demo"]
    frames.sort(key=lambda f: f.get("frame_index", 0))
    states = np.array([f["state"] for f in frames], dtype=float)
    return frames, states


def keyframes(states):
    w = states[:, 6] + states[:, 7]
    w0 = w[:5].max()
    closing = np.where(w < w0 - 0.008)[0]
    if len(closing) == 0:
        raise RuntimeError("no grasp found in video")
    gi = int(closing[0])
    # grasp pose: the pose when the gripper started closing (lowest settled point)
    pre = max(0, gi - 1)
    grasp = states[pre]
    # post-grasp waypoints: subsample by motion
    post = []
    last = states[gi]
    for i in range(gi, len(states)):
        s = states[i]
        dp = np.linalg.norm(s[:3] - last[:3])
        dr = np.degrees((R.from_rotvec(s[3:6]) * R.from_rotvec(last[3:6]).inv()).magnitude())
        if dp > 0.015 or dr > 6:
            post.append((i, s.copy()))
            last = s
    post.append((len(states) - 1, states[-1].copy()))
    wclosed = float(np.median(w[gi + 10:])) if len(w) > gi + 10 else float(w[-1])
    return gi, grasp, post, wclosed


def wrist_measure(depth_path, K, pose, state):
    """In-hand peg measurement in the gripper's planar frame: yaw offset and centroid offset."""
    from PIL import Image
    from perception import peg_line
    dep = np.array(Image.open(WS / depth_path)).astype(float) / 1000.0
    L = peg_line(dep, K, pose)
    if L is None:
        return None
    s = np.asarray(state, dtype=float)
    gyaw = R.from_rotvec(s[3:6]).as_euler("xyz")[2]
    rel = L["center"] - s[:3]
    c, sn = np.cos(-gyaw), np.sin(-gyaw)
    o = np.array([c * rel[0] - sn * rel[1], sn * rel[0] + c * rel[1], rel[2]])
    dyaw = np.radians(L["yaw"]) - gyaw
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    return {"o": o, "dyaw": float(dyaw), "n": L["n"], "len": L["len"], "yaw": L["yaw"]}


def frame_files(f):
    return {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}


def video_inhand(frames, states, gi):
    """Average in-hand measurement over stationary, lifted video frames after the grasp."""
    out = []
    for i in range(gi + 5, len(states) - 1):
        if states[i, 2] < states[gi, 2] + 0.08:
            continue
        if np.linalg.norm(states[i + 1, :3] - states[i, :3]) > 0.0003 or np.linalg.norm(states[i, :3] - states[i - 1, :3]) > 0.0003:
            continue
        fl = frame_files(frames[i])
        if ("wrist", "depth") not in fl:
            continue
        m = wrist_measure(fl[("wrist", "depth")], frames[i]["camera_intrinsics"]["wrist"]["matrix"],
                          frames[i]["camera_poses"]["wrist"], states[i])
        if m is not None and m["n"] > 1000:
            out.append(m)
        if len(out) >= 6:
            break
    if not out:
        return None
    return {"o": np.median([m["o"] for m in out], axis=0), "dyaw": float(np.median([m["dyaw"] for m in out])),
            "n": len(out), "len": float(np.median([m["len"] for m in out]))}


def live_inhand(rb):
    """Measure the in-hand peg from the live wrist depth attachment of a fresh observation."""
    import base64, io
    from PIL import Image
    from perception import peg_line
    reply = rb.ctx.call("robomme_observe")
    p = packet(reply)
    rb._absorb(p)
    try:
        idx = None
        for a in p.get("attachments", []):
            if a.get("camera") == "wrist" and a.get("kind") == "depth":
                idx = a["content_index"]
        blk = reply["content"][idx]
        dep = np.array(Image.open(io.BytesIO(base64.b64decode(blk["data"])))).astype(float) / 1000.0
        L = peg_line(dep, p["camera_intrinsics"]["wrist"]["matrix"], p["camera_poses"]["wrist"])
        if L is None:
            return None
        s = np.asarray(p["state"], dtype=float)
        gyaw = R.from_rotvec(s[3:6]).as_euler("xyz")[2]
        rel = L["center"] - s[:3]
        c, sn = np.cos(-gyaw), np.sin(-gyaw)
        o = np.array([c * rel[0] - sn * rel[1], sn * rel[0] + c * rel[1], rel[2]])
        dyaw = np.radians(L["yaw"]) - gyaw
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        return {"o": o, "dyaw": float(dyaw), "n": L["n"], "len": L["len"], "yaw": L["yaw"]}
    except Exception as ex:
        log("live_inhand failed", repr(ex))
        return None


def stroke(states):
    """Final insertion stroke: walk back from the end while the motion stays straight and level."""
    end = states[-1, :3]
    j = len(states) - 1
    far = None
    for i in range(len(states) - 1, -1, -1):
        d = end - states[i, :3]
        if abs(states[i, 2] - end[2]) > 0.004:
            break
        if np.linalg.norm(d[:2]) > 0.02:
            far = i
        j = i
    if far is None:
        far = j
    k = j
    for i in range(j, len(states)):
        if np.linalg.norm(end[:2] - states[i, :2]) < 0.07:
            k = i
            break
    axis = end - states[k, :3]
    axis[2] = 0.0
    axis /= max(1e-9, np.linalg.norm(axis))
    return j, axis


def yaw_of(rot):
    return rot.as_euler("xyz")[2]


def rotz(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def decode_depth(reply, p, camera="wrist"):
    import base64, io
    from PIL import Image
    for a in p.get("attachments", []):
        if a.get("camera") == camera and a.get("kind") == "depth":
            blk = reply["content"][a["content_index"]]
            return np.array(Image.open(io.BytesIO(base64.b64decode(blk["data"])))).astype(float) / 1000.0
    return None


def peg_world(dep, K, pose, axis, tcp):
    """World-frame peg features from wrist depth, restricted to a box around the expected in-hand peg:
    lateral offset of the top-face centroid (perp. to axis), top-face z, and horizontal yaw."""
    from perception import backproject
    vs, us = np.nonzero((dep > 0.02) & (dep < 0.2))
    pts = backproject(dep, K, pose, us, vs)
    rel = pts - np.asarray(tcp)[None, :3]
    latv = np.array([-axis[1], axis[0]])
    al = rel[:, :2] @ axis[:2]
    la = rel[:, :2] @ latv
    sel = (al > 0.015) & (al < 0.07) & (np.abs(la) < 0.015) & (np.abs(rel[:, 2]) < 0.016)
    pts = pts[sel]
    if len(pts) < 500:
        return None
    z90 = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > z90 - 0.003]
    c = top.mean(0)
    xy = top[:, :2] - top[:, :2].mean(0)
    w, V = np.linalg.eigh(xy.T @ xy)
    d = V[:, -1]
    if np.dot(d, axis[:2]) < 0:
        d = -d
    return {"lat": float(np.dot(c[:2], latv)), "along": float(np.dot(c[:2], axis[:2])), "ztop": float(z90),
            "yaw": float(np.arctan2(d[1], d[0])), "n": int(len(pts)), "ntop": int(len(top)), "c": c}


def video_peg_world(frame, axis):
    from PIL import Image
    fl = frame_files(frame)
    dep = np.array(Image.open(WS / fl[("wrist", "depth")])).astype(float) / 1000.0
    return peg_world(dep, frame["camera_intrinsics"]["wrist"]["matrix"], frame["camera_poses"]["wrist"], axis, frame["state"][:3])


def live_peg_world(rb, axis):
    reply = rb.ctx.call("robomme_observe")
    p = packet(reply)
    rb._absorb(p)
    dep = decode_depth(reply, p)
    if dep is None:
        return None
    return peg_world(dep, p["camera_intrinsics"]["wrist"]["matrix"], p["camera_poses"]["wrist"], axis, p["state"][:3])


def T_to_pose(T):
    return T[:3, 3].copy(), R.from_matrix(T[:3, :3])


def video_H(frames, states, k, axis):
    from PIL import Image
    from perception import peg_frame, tcp_matrix
    out = []
    for kk in [k - 4, k - 2, k]:
        if kk < 0:
            continue
        f = frames[kk]
        fl = frame_files(f)
        dep = np.array(Image.open(WS / fl[("wrist", "depth")])).astype(float) / 1000.0
        r = peg_frame(dep, f["camera_intrinsics"]["wrist"]["matrix"], f["camera_poses"]["wrist"], states[kk], axis)
        if r is not None:
            out.append(np.linalg.inv(tcp_matrix(states[kk])) @ r[0])
    return out[-1] if out else None


def live_H(rb, axis):
    from perception import peg_frame, tcp_matrix
    reply = rb.ctx.call("robomme_observe")
    p = packet(reply)
    rb._absorb(p)
    dep = decode_depth(reply, p)
    if dep is None:
        return None
    r = peg_frame(dep, p["camera_intrinsics"]["wrist"]["matrix"], p["camera_poses"]["wrist"], p["state"], axis)
    if r is None:
        return None
    return np.linalg.inv(tcp_matrix(p["state"])) @ r[0]


def H_str(H):
    return "pos(mm) %s eul(deg) %s" % (np.round(H[:3, 3] * 1000, 2).tolist(), np.round(R.from_matrix(H[:3, :3]).as_euler("xyz", degrees=True), 2).tolist())


def main(ctx):
    from perception import tcp_matrix
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rb = Robot(ctx)
    p0 = rb.observe()
    frames, states = load_video(p0)
    log("v9 video frames", len(frames), "exec state", np.round(rb.state, 4).tolist(), "rem", rb.remaining)
    gi, grasp, post, wclosed = keyframes(states)
    j, axis = stroke(states)
    end = states[-1].copy()
    nv = len(states)
    STANDOFF = 0.056
    ka = j
    for i in range(j, nv):
        if np.linalg.norm(end[:2] - states[i, :2]) <= STANDOFF:
            ka = i
            break
    log("grasp idx", gi, "grasp pose", np.round(grasp, 4).tolist(), "closed width", round(wclosed, 4))
    log("stroke start", j, "align frame", ka, np.round(states[ka, :3], 4).tolist(), "end", np.round(end[:3], 4).tolist(),
        "axis", np.round(axis, 3).tolist())
    Hv = video_H(frames, states, ka, axis)
    log("video H", None if Hv is None else H_str(Hv))

    gpos, grot = grasp[:3].copy(), R.from_rotvec(grasp[3:6])
    # 1. grasp exactly where the video grasped
    log("move above", rb.move_to(gpos + np.array([0, 0, 0.10]), grot, -1, tol=0.004, max_steps=80))
    log("descend", rb.servo(gpos + np.array([0, 0, 0.03]), grot, -1, tol=0.002, max_steps=50, vmax=0.4))
    log("descend2", rb.servo(gpos, grot, -1, tol=0.0003, max_steps=50, vmax=0.08))
    log("pos at grasp", np.round(rb.state, 4).tolist())
    Tgl = tcp_matrix(rb.state)
    gp = gpos.copy()
    closed_at = None
    for t in range(50):
        if rb.ended:
            break
        if t >= 20:  # fingers slow to close (e.g. dragging on the table): rise slowly while closing
            gp = gpos + np.array([0, 0, min(0.004, (t - 19) * 0.0002)])
        rb.hold_bias(1, 1, gp, grot)
        if rb.width < 0.022:
            Tgl = tcp_matrix(rb.state)
            closed_at = t
            break
    rb.hold_bias(1, 6, gp, grot)
    log("closed at step", closed_at, "width", round(rb.width, 4), "tcp z", round(float(rb.pos[2]), 4))
    Dg = np.linalg.inv(tcp_matrix(grasp)) @ Tgl
    Ck = np.eye(4)
    Ck[0, 3], Ck[2, 3] = Dg[0, 3], Dg[2, 3]
    log("grasp offset in tcp frame mm", np.round(Dg[:3, 3] * 1000, 2).tolist(), "rot deg",
        round(float(np.degrees(R.from_matrix(Dg[:3, :3]).magnitude())), 3))
    rb.bias[:] = 0
    log("after close width", round(rb.width, 4))
    # 2. transport through the video's waypoints up to the stroke
    trans = [(i, s) for i, s in post if i < j]
    for k, (i, s) in enumerate(trans):
        if rb.ended:
            break
        r = rb.move_to(s[:3], R.from_rotvec(s[3:6]), 1, tol=0.005, tol_deg=3.0, max_steps=40, vmax=0.6)
        if k % 6 == 0:
            log("wp", k, i, np.round(s[:3], 3).tolist(), "err", np.round(r[:2], 4).tolist(), "w", round(rb.width, 4))
    lat = np.array([-axis[1], axis[0], 0.0])
    Tv = [tcp_matrix(s) for s in states]
    C = np.eye(4)

    def target(i, extra=np.zeros(3)):
        T = Tv[i] @ C
        pos, rot = T_to_pose(T)
        return pos + extra, rot

    def settle(pos, rot, n_ok=3, max_steps=40):
        ok = 0
        for _ in range(max_steps):
            if rb.ended:
                break
            e = pos - rb.pos
            ed = np.degrees((rot * rb.rot.inv()).magnitude())
            if np.linalg.norm(e) < 0.0003 and ed < 0.25:
                ok += 1
                if ok >= n_ok:
                    break
            else:
                ok = 0
            rb.bias = np.clip(rb.bias + 0.12 * e, -0.004, 0.004)
            dp = np.clip((e + rb.bias) / 0.05, -0.06, 0.06)
            dr = np.clip((rot * rb.rot.inv()).as_rotvec() / 0.5, -0.1, 0.1)
            rb.step([list(dp) + list(dr) + [1]])
        return rb.err(pos, rot)

    def calibrate():
        nonlocal C
        if Hv is None:
            return
        for it in range(3):
            tp, tr = target(ka)
            r = settle(tp, tr)
            Hm = live_H(rb, axis)
            if Hm is None:
                log("calib: no live H")
                return
            Cn = Hv @ np.linalg.inv(Hm)
            dpos = np.linalg.norm(Cn[:3, 3] - C[:3, 3])
            drot = np.degrees(R.from_matrix(Cn[:3, :3] @ C[:3, :3].T).magnitude())
            log("calib it", it, "settle err", np.round(r[:2], 4).tolist(), "live H", H_str(Hm), "C", H_str(Cn),
                "change mm/deg", round(dpos * 1000, 2), round(drot, 2))
            if np.linalg.norm(Cn[:3, 3]) > 0.01 or np.degrees(R.from_matrix(Cn[:3, :3]).magnitude()) > 8:
                log("calib: implausible correction, keep previous")
                return
            C = Cn
            if dpos < 0.0002 and drot < 0.15:
                break
        tp, tr = target(ka)
        settle(tp, tr)

    def dist_end(i):
        return float(np.linalg.norm(end[:2] - states[i, :2]))

    def frame_at(d):
        for i in range(ka, nv):
            if dist_end(i) <= d:
                return i
        return ka

    def rot_about(Cb, ax, deg, pivot):
        M = np.eye(4)
        Rm = R.from_rotvec(np.radians(deg) * np.asarray(ax, dtype=float)).as_matrix()
        M[:3, :3] = Rm
        M[:3, 3] = pivot - Rm @ pivot
        return Cb @ M

    def run_stroke(k0, extra, tag):
        lag_n = 0
        for i in range(k0, nv):
            if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 5):
                return False, i
            tp, tr = target(i, extra)
            e = tp - rb.pos
            rb.bias = np.clip(rb.bias + 0.2 * e, -0.004, 0.004)
            dp = np.clip((e + rb.bias) / 0.05, -0.3, 0.3)
            dr = np.clip((tr * rb.rot.inv()).as_rotvec() / 0.5, -0.2, 0.2)
            rb.step([list(dp) + list(dr) + [1]])
            behind = float(np.dot(tp - rb.pos, axis))
            if (i - k0) % 8 == 0:
                log("stroke", tag, i, "behind mm", round(behind * 1000, 2), "latdev mm", round(float(np.dot(rb.pos - tp, lat)) * 1000, 2),
                    "zdev mm", round(float(rb.pos[2] - tp[2]) * 1000, 2), "rotdev", round(rb.err(tp, tr)[1], 2), "w", round(rb.width, 4),
                    "complete", rb.complete, "rem", rb.remaining)
            lag_n = lag_n + 1 if behind > 0.0025 else 0
            if lag_n >= 8:
                log("stall", tag, "at frame", i, "dist_end", round(dist_end(i), 4), "pos", np.round(rb.pos, 4).tolist())
                return False, i
        return True, nv - 1

    def nearest_frame(pos):
        best, bi = 1e9, ka
        for i in range(ka, nv):
            d = abs(float(np.dot(target(i)[0] - pos, axis)))
            if d < best:
                best, bi = d, i
        return bi

    def push_through(tag, fwd, amp, n=40, patience=15, extra=np.zeros(3)):
        """Push along the axis with lateral/vertical P-correction to the video line plus an optional circular wiggle."""
        s_best = float(np.dot(rb.pos, axis))
        s_start = s_best
        idle = 0
        for t in range(n):
            if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 5):
                break
            i = nearest_frame(rb.pos)
            if i >= nv - 2:
                log("push", tag, "reached end frames")
                return True
            tp, tr = target(min(i + 2, nv - 1), extra)
            e = tp - rb.pos
            el = e - axis * float(np.dot(e, axis))
            off = amp * (lat * np.cos(t * 0.9) + np.array([0, 0, 1.0]) * np.sin(t * 0.9))
            dp = np.clip(axis * fwd + (el + off) / 0.05, -1, 1)
            dr = np.clip((tr * rb.rot.inv()).as_rotvec() / 0.5, -0.2, 0.2)
            rb.step([list(dp) + list(dr) + [1]])
            s = float(np.dot(rb.pos, axis))
            if s > s_best + 0.0005:
                s_best, idle = s, 0
            else:
                idle += 1
            if idle >= patience:
                break
        log("push", tag, "fwd", fwd, "amp mm", amp * 1000, "advanced mm", round((s_best - s_start) * 1000, 2),
            "frame", nearest_frame(rb.pos), "complete", rb.complete, "rem", rb.remaining)
        return rb.complete

    def escalate(tag, extra=np.zeros(3)):
        for fwd, amp in [(0.25, 0.0), (0.15, 0.0003)]:
            if rb.ended or rb.complete:
                break
            if push_through(tag, fwd, amp, extra=extra):
                return True
        return rb.complete

    zhat = np.array([0, 0, 1.0])

    def spiral(tag, extra, rmax=0.0022, pitch=0.0003, ds=0.00035, press=0.0015, max_steps=190):
        """Light-force spiral search in the lateral/vertical plane at the hole face; returns the offset that dropped in."""
        i0 = nearest_frame(rb.pos)
        tp0, tr0 = target(i0, extra)
        s_c = float(np.dot(rb.pos, axis))
        base = tp0 + axis * (s_c - float(np.dot(tp0, axis)))
        theta, n = 0.0, 0
        bl = np.zeros(3)
        while not rb.ended and not rb.complete and n < max_steps and (rb.remaining is None or rb.remaining > 30):
            r = pitch * theta / (2 * np.pi)
            if r > rmax:
                break
            off = r * (np.cos(theta) * lat + np.sin(theta) * zhat)
            goal = base + off + axis * press
            e = goal - rb.pos
            ea = float(np.dot(e, axis))
            el = e - axis * ea
            bl = np.clip(bl + 0.3 * el, -0.003, 0.003)
            bl = bl - axis * float(np.dot(bl, axis))
            dp = np.clip((el + bl) / 0.05 + axis * np.clip(ea / 0.05, -0.1, 0.06), -0.3, 0.3)
            dr = np.clip((tr0 * rb.rot.inv()).as_rotvec() / 0.5, -0.1, 0.1)
            rb.step([list(dp) + list(dr) + [1]])
            s = float(np.dot(rb.pos, axis))
            if n % 20 == 0:
                log("spiral", tag, n, "r mm", round(r * 1000, 2), "adv mm", round((s - s_c) * 1000, 2), "rem", rb.remaining)
            if s - s_c > 0.003:
                log("spiral", tag, "DROP-IN at step", n, "off mm", np.round(off * 1000, 2).tolist(), "adv mm", round((s - s_c) * 1000, 2))
                return off
            theta += ds / max(r, 0.0006)
            n += 1
        log("spiral", tag, "no drop-in after", n, "steps")
        return None

    def compliant(tag, fwd=0.12, ramp=0.0, n=60, patience=20):
        """Forward-only push along the axis: zero lateral/rotational delta so the hole can guide the peg.
        ramp > 0 adds a small oscillating rotation (rad/step command units)."""
        s0 = float(np.dot(rb.pos, axis))
        s_end = float(np.dot(target(nv - 1)[0], axis))
        best, idle = s0, 0
        for t in range(n):
            if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 5):
                break
            dr = np.zeros(3)
            if ramp > 0:
                dr = ramp * (np.cos(t * 0.7) * lat + np.sin(t * 0.7) * zhat)
            rb.step([list(axis * fwd) + list(dr) + [1]])
            s = float(np.dot(rb.pos, axis))
            if s > best + 0.0003:
                best, idle = s, 0
            else:
                idle += 1
            if t % 10 == 0:
                log("compliant", tag, t, "adv mm", round((s - s0) * 1000, 2), "to end mm", round((s_end - s) * 1000, 2),
                    "complete", rb.complete, "rem", rb.remaining)
            if s > s_end + 0.004 or idle >= patience:
                break
        log("compliant", tag, "done adv mm", round((best - s0) * 1000, 2), "to end mm", round((s_end - best) * 1000, 2), "complete", rb.complete)
        return best - s0

    def line_point(s, extra):
        best, bi = 1e9, ka
        for i in range(ka, nv):
            d = abs(float(np.dot(target(i, extra)[0], axis)) - s)
            if d < best:
                best, bi = d, i
        tp, tr = target(bi, extra)
        return tp + axis * (s - float(np.dot(tp, axis))), tr

    def try_offset(off, s0, s_stall, extra, n=22, v=0.0006):
        p, r = line_point(s0, extra)
        rb.servo(p + off, r, 1, tol=0.0003, max_steps=14, vmax=0.08)
        lag = 0
        for k in range(n):
            if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 10):
                break
            p, r = line_point(s0 + v * (k + 1), extra)
            p = p + off
            e = p - rb.pos
            rb.bias = np.clip(rb.bias + 0.2 * e, -0.003, 0.003)
            dp = np.clip((e + rb.bias) / 0.05, -0.3, 0.3)
            dr = np.clip((r * rb.rot.inv()).as_rotvec() / 0.5, -0.1, 0.1)
            rb.step([list(dp) + list(dr) + [1]])
            s = float(np.dot(rb.pos, axis))
            if s > s_stall + 0.002:
                return True
            lag = lag + 1 if float(np.dot(p - rb.pos, axis)) > 0.0012 else 0
            if lag >= 3:
                break
        return rb.complete or float(np.dot(rb.pos, axis)) > s_stall + 0.002

    def grid_search(tag, extra):
        s_stall = float(np.dot(rb.pos, axis))
        s0 = s_stall - 0.0045
        offs = []
        for rr in [0.0005, 0.001, 0.0015]:
            for a in range(8):
                th = a * np.pi / 4 + (np.pi / 8 if rr == 0.001 else 0.0)
                offs.append(rr * (np.cos(th) * lat + np.sin(th) * zhat))
        for n_try, off in enumerate(offs):
            if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 45):
                break
            ok_ = try_offset(off, s0, s_stall, extra)
            log("grid", tag, n_try, "off lat/z mm", round(float(np.dot(off, lat)) * 1000, 2), round(float(off[2]) * 1000, 2),
                "adv mm", round((float(np.dot(rb.pos, axis)) - s_stall) * 1000, 2), "in" if ok_ else "blocked", "rem", rb.remaining)
            if ok_:
                return off
            p, r = line_point(s0 - 0.001, extra)
            rb.servo(p + off, r, 1, tol=0.0006, max_steps=8, vmax=0.08)
        return None

    tip = np.array([0.08, 0.0, 0.0])
    z = np.zeros(3)
    C = Ck.copy()
    tp, tr = target(ka)
    rb.servo(tp - axis * 0.01 + np.array([0, 0, 0.003]), tr, 1, tol=0.0015, max_steps=50, vmax=0.3)
    if Hv is not None:
        for it in range(2):
            tp, tr = target(ka)
            settle(tp, tr)
            Hm = live_H(rb, axis)
            if Hm is None:
                break
            Cn = Hv @ np.linalg.inv(Hm)
            dt = Cn[:3, 3] - Ck[:3, 3]
            log("H check", it, "live H", H_str(Hm), "Cn", H_str(Cn), "diff vs kin mm", np.round(dt * 1000, 2).tolist())
            if 0.002 < np.linalg.norm(dt) < 0.015:
                Ck = np.eye(4)
                Ck[:3, 3] = Cn[:3, 3]
                C = Ck.copy()
                log("using translation-only H correction", np.round(Ck[:3, 3] * 1000, 2).tolist())
            else:
                break
    plans = [("kin", Ck, z), ("kin_z+0.4", Ck, np.array([0, 0, 0.0004])),
             ("roll+1.5", rot_about(Ck, [1, 0, 0], 1.5, tip), z), ("roll-1.5", rot_about(Ck, [1, 0, 0], -1.5, tip), z),
             ("H", None, z), ("yaw+0.8", rot_about(Ck, [0, 0, 1], 0.8, tip), z), ("yaw-0.8", rot_about(Ck, [0, 0, 1], -0.8, tip), z),
             ("pitch+0.8", rot_about(Ck, [0, 1, 0], 0.8, tip), z), ("pitch-0.8", rot_about(Ck, [0, 1, 0], -0.8, tip), z),
             ("kin_z-0.4", Ck, np.array([0, 0, -0.0004]))]
    k0 = ka
    for pi, (tag, Cp, extra) in enumerate(plans):
        if rb.ended or rb.complete or (rb.remaining is not None and rb.remaining < 40):
            break
        if Cp is None:
            calibrate()
        else:
            C = Cp
        tp, tr = target(k0, extra)
        rb.servo(tp, tr, 1, tol=0.001, max_steps=40, vmax=0.06)
        r = settle(tp, tr)
        log("plan", tag, "start frame", k0, "settle err", np.round(r[:2], 4).tolist(), "C", H_str(C), "rem", rb.remaining)
        ok, si = run_stroke(k0, extra, tag)
        if ok and not rb.ended:
            tp, tr = target(nv - 1, extra)
            r = rb.servo(tp, tr, 1, tol=0.0005, max_steps=25, vmax=0.05)
            log("end reached", np.round(r[:2], 4).tolist(), "complete", rb.complete)
            if not rb.complete and not rb.ended:
                r = rb.servo(tp + axis * 0.004, tr, 1, tol=0.0005, max_steps=20, vmax=0.03)
                log("deeper", np.round(r[:2], 4).tolist(), "complete", rb.complete)
            break
        if rb.ended or rb.complete:
            break
        if pi == 0 and not rb.complete:
            try:
                off = grid_search(tag, extra)
            except Exception as ex:  # never let the search crash the run
                log("grid_search error", repr(ex))
                off = None
            if off is not None and not rb.ended and not rb.complete:
                e2 = extra + off
                k1 = nearest_frame(rb.pos)
                ok, si = run_stroke(k1, e2, tag + "_grid_in")
                extra = e2
                if ok or rb.complete:
                    tp, tr = target(nv - 1, extra)
                    r = rb.servo(tp, tr, 1, tol=0.0005, max_steps=25, vmax=0.05)
                    log("end after grid", np.round(r[:2], 4).tolist(), "complete", rb.complete)
                    if not rb.complete and not rb.ended:
                        r = rb.servo(tp + axis * 0.004, tr, 1, tol=0.0005, max_steps=20, vmax=0.03)
                        log("deeper", np.round(r[:2], 4).tolist(), "complete", rb.complete)
                    break
                if not rb.ended and not rb.complete:
                    push_through(tag + "_grid_push", 0.25, 0.0, extra=extra)
        if rb.complete or rb.ended:
            break
        if False and pi < 2 and not rb.complete:
            for fwd, ramp in [(0.12, 0.0), (0.2, 0.0), (0.12, 0.02), (0.25, 0.04)]:
                adv = compliant(tag + "_c%.2f_%.2f" % (fwd, ramp), fwd, ramp)
                if rb.complete or rb.ended:
                    break
                if adv > 0.003:  # moving in: keep pushing compliantly
                    for _ in range(4):
                        if compliant(tag + "_more", fwd, 0.0) < 0.001 or rb.complete or rb.ended:
                            break
        if rb.complete or rb.ended:
            break
        if pi == 1:
            off = spiral(tag, extra)
            if off is not None and not rb.ended:
                e2 = extra + off
                k1 = nearest_frame(rb.pos)
                ok, si = run_stroke(k1, e2, tag + "_in")
                if not ok and not rb.ended and not rb.complete:
                    escalate(tag + "_in", e2)
                extra = e2
        if ok or rb.ended or rb.complete:
            tp, tr = target(nv - 1, extra)
            r = rb.servo(tp, tr, 1, tol=0.0005, max_steps=25, vmax=0.05)
            log("end after push", np.round(r[:2], 4).tolist(), "complete", rb.complete)
            break
        # back off along the axis, then restart from ~2 cm before the stall point
        rb.servo(rb.pos - axis * 0.008, rb.rot, 1, tol=0.0015, max_steps=15, vmax=0.06)
        k0 = frame_at(dist_end(si) + 0.02)
    if not rb.ended and not rb.complete:
        rb.hold(1, 10)
    log("done; complete", rb.complete, "ended", rb.ended, "rem", rb.remaining, "t", round(time.time() - t0, 1))
    log("final state", np.round(rb.state, 4).tolist())
