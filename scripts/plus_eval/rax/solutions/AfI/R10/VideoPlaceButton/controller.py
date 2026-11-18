"""VideoPlaceButton controller (v1).

Strategy: parse the conditioning-video manifest (robot state per frame) into
grasp / release / button-press events, select the target placement named by the
instruction, then pick the cube (at its last release position, refined by
vision) and place it on that target.
"""
import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path

import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WS = Path("/workspace")
LOG_DIR = WS / "output"
_LOG = []
RELEASE_DZ = -0.001
AFTER_MODE = "first_after"


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    _LOG.append(msg)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "vpb_log.txt", "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- MCP helpers
def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                pass
    return {}


def images(reply, pk):
    out = {}
    blocks = reply.get("content", [])
    for att in pk.get("attachments", []):
        idx = att.get("content_index")
        if idx is None or idx >= len(blocks):
            continue
        b = blocks[idx]
        data = b.get("data")
        if not data:
            continue
        from PIL import Image
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        arr = np.array(im)
        out[(att["camera"], att["kind"])] = arr
    return out


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.complete = False
        self.remaining = 10 ** 6
        self.step_id = 0

    def observe(self):
        r = self.ctx.call("robomme_observe")
        pk = packet(r)
        self._upd(pk)
        return r, pk

    def _upd(self, pk):
        if "state" in pk:
            self.state = np.array(pk["state"], dtype=float)
        lim = pk.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        if pk.get("episode_ended"):
            self.ended = True
        if "step_id" in pk:
            self.step_id = pk["step_id"]
        for s in pk.get("steps", []) or []:
            if isinstance(s, dict):
                if s.get("task_complete"):
                    self.complete = True
                if s.get("episode_ended"):
                    self.ended = True
        if pk.get("task_complete"):
            self.complete = True
        tr = pk.get("transition") or {}
        if tr.get("task_complete"):
            self.complete = True
        if tr.get("episode_ended"):
            self.ended = True
        for s in tr.get("steps", []) or []:
            if isinstance(s, dict) and s.get("task_complete"):
                self.complete = True

    def step(self, rows):
        if self.ended:
            return None
        rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
        out = None
        for i in range(0, len(rows), 100):
            if self.ended:
                break
            chunk = rows[i:i + 100]
            if self.remaining is not None and len(chunk) > self.remaining:
                chunk = chunk[:max(0, self.remaining)]
                if not chunk:
                    self.ended = True
                    break
            r = self.ctx.call("robomme_step", actions=chunk)
            pk = packet(r)
            self._upd(pk)
            out = (r, pk)
        return out


# ---------------------------------------------------------------- rotations
def rotvec_to_mat(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def mat_to_rotvec(R):
    c = (np.trace(R) - 1) / 2
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    if th < 1e-7:
        return np.zeros(3)
    if th > math.pi - 1e-4:
        # near pi: axis from diagonal
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(A), 0, None))
        i = int(np.argmax(k))
        if k[i] > 1e-6:
            k = A[:, i] / k[i]
        k = k / np.linalg.norm(k)
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def rot_err(cur_rv, tgt_rv):
    """World-frame rotvec taking current to target."""
    return mat_to_rotvec(rotvec_to_mat(tgt_rv) @ rotvec_to_mat(cur_rv).T)


def yaw_of(rv):
    R = rotvec_to_mat(rv)
    # gripper x-axis projected to the table
    return math.atan2(R[1, 0], R[0, 0])


def down_with_yaw(yaw):
    """Gripper pointing down (z-axis -world z) with given yaw of its x axis."""
    Rd = rotvec_to_mat([math.pi, 0, 0])  # reference down pose (x along +x)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return mat_to_rotvec(Rz @ Rd)


# ---------------------------------------------------------------- motion
def move_to(robot, xyz, rv=None, grip=-1.0, speed=0.6, tol=0.004, max_iter=6, settle=2):
    """Closed-loop chunked motion to xyz (and orientation rv)."""
    xyz = np.asarray(xyz, float)
    for it in range(max_iter):
        if robot.ended or robot.state is None:
            return False
        p = robot.state[:3]
        d = xyz - p
        dist = float(np.linalg.norm(d))
        re_ = rot_err(robot.state[3:6], rv) if rv is not None else np.zeros(3)
        ang = float(np.linalg.norm(re_))
        if dist < tol and ang < 0.03:
            return True
        n_pos = math.ceil(dist / (0.05 * speed)) if dist > 0 else 0
        n_rot = math.ceil(ang / (0.5 * 0.5)) if ang > 0 else 0
        n = max(1, n_pos, n_rot)
        if it > 0:
            n = max(1, min(n, 8))
        rows = []
        for k in range(n):
            dp = d / n / 0.05
            dr = re_ / n / 0.5
            rows.append(list(dp) + list(dr) + [grip])
        rows += [[0, 0, 0, 0, 0, 0, grip]] * settle
        robot.step(rows)
    return False


def gripper_cmd(robot, grip, n=10):
    robot.step([[0, 0, 0, 0, 0, 0, grip]] * n)


# ---------------------------------------------------------------- video parse
def load_manifest(pk):
    seq = pk.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        return None
    p = WS / mp
    if not p.exists():
        return None
    return json.loads(p.read_text())


def parse_events(frames):
    """Return list of events (kind, frame_idx, xyz, rotvec)."""
    vf = [f for f in frames if f.get("phase") == "video_demo" and f.get("state")]
    vf.sort(key=lambda f: f["frame_index"])
    if not vf:
        return []
    S = np.array([f["state"] for f in vf], float)
    g = S[:, 6] + S[:, 7]
    z = S[:, 2]
    # gripper class: 0 open (>0.07), 1 holding (0.01..0.07), 2 closed empty (<0.01)
    cls = np.where(g > 0.07, 0, np.where(g > 0.012, 1, 2))
    events = []
    n = len(vf)
    # holding segments: grasp at start, release at end
    i = 0
    while i < n:
        if cls[i] == 1:
            j = i
            while j + 1 < n and cls[j + 1] == 1:
                j += 1
            # segment i..j holding (ignore short transitional blips)
            if j - i >= 3:
                gi = i
                events.append(("grasp", vf[gi]["frame_index"], S[gi, :3].copy(), S[gi, 3:6].copy()))
                # release: frame where gripper opens after j; use position at j
                if j + 1 < n and cls[j + 1] == 0 or (j + 1 < n and g[min(n - 1, j + 2)] > 0.06):
                    events.append(("release", vf[j]["frame_index"], S[j, :3].copy(), S[j, 3:6].copy()))
                elif j + 1 >= n:
                    events.append(("held_end", vf[j]["frame_index"], S[j, :3].copy(), S[j, 3:6].copy()))
                else:
                    events.append(("release", vf[j]["frame_index"], S[j, :3].copy(), S[j, 3:6].copy()))
            i = j + 1
        elif cls[i] == 2:
            j = i
            while j + 1 < n and cls[j + 1] == 2:
                j += 1
            seg = np.arange(i, j + 1)
            k = seg[int(np.argmin(z[seg]))]
            if j - i >= 2 and z[k] < 0.10:
                events.append(("button", vf[k]["frame_index"], S[k, :3].copy(), S[k, 3:6].copy()))
            i = j + 1
        else:
            i += 1
    events.sort(key=lambda e: e[1])
    return events


def build_chains(events, tol=0.03):
    """Group grasp/release events into per-object chains by position continuity.
    Each chain: dict(start=xy, events=[...], end=xy)."""
    chains = []
    holding = None
    for e in events:
        if e[0] == "grasp":
            xy = e[2][:2]
            best, bd = None, tol
            for c in chains:
                d = float(np.linalg.norm(c["end"] - xy))
                if d < bd and not c.get("held"):
                    best, bd = c, d
            if best is None:
                best = {"start": xy.copy(), "end": xy.copy(), "events": []}
                chains.append(best)
            best["held"] = True
            best["events"].append(e)
            holding = best
        elif e[0] in ("release", "held_end"):
            if holding is not None:
                holding["events"].append(e)
                holding["end"] = e[2][:2].copy()
                holding["held"] = False
                holding = None
    return chains


def choose_target(events, instruction):
    ins = instruction.lower()
    places = [e for e in events if e[0] == "release"]
    buttons = [e for e in events if e[0] == "button"]
    if not places:
        return None, "no placements"
    if not buttons:
        return places[-1], "no button; last placement"
    bf = buttons[0][1]
    before = re.search(r"\bbefore\b", ins) is not None
    after = re.search(r"\bafter\b", ins) is not None
    first = re.search(r"\bfirst\b", ins) is not None
    last = re.search(r"\b(last|final|latest)\b", ins) is not None
    if before and not after:
        cand = [p for p in places if p[1] < bf]
        if not cand:
            return places[0], "before: none, fallback first"
        if first:
            return cand[0], "first before button"
        return cand[-1], "last before button"
    if AFTER_MODE == "resting_at_press":
        cand = [p for p in places if p[1] < bf]
        if cand:
            return cand[-1], "after(resting_at_press): last release before button"
    cand = [p for p in places if p[1] > bf]
    if not cand:
        return places[-1], "after: none, fallback last"
    if last:
        return cand[-1], "last after button"
    return cand[0], "first after button"


def parse_color(instruction):
    ins = instruction.lower()
    for c in ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black", "cyan"]:
        if re.search(r"\b%s\b" % c, ins):
            return c
    return "red"


# ---------------------------------------------------------------- vision
def cam_points(depth_mm, intr, pose, mask):
    K = np.array(intr["matrix"], float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = np.nonzero(mask)
    d = depth_mm[vs, us].astype(float) * 0.001
    ok = d > 0
    vs, us, d = vs[ok], us[ok], d[ok]
    pc = np.stack([(us + 0.5 - cx) * d / fx, -(vs + 0.5 - cy) * d / fy, -d], 1)
    w, x, y, z = pose["quaternion_wxyz"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return pc @ R.T + np.array(pose["position_m"], float)


def color_mask(rgb, color):
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    if color == "red":
        return (r > 120) & (g < 60) & (b < 60)
    if color == "green":
        return (g > 100) & (r < 80) & (b < 80)
    if color == "blue":
        return (b > 100) & (r < 60) & (g < 60)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 60)
    return (r > 120) & (g < 60) & (b < 60)


def locate_cube(reply, pk, color, cam="front"):
    ims = images(reply, pk)
    rgb = ims.get((cam, "rgb"))
    dep = ims.get((cam, "depth"))
    if rgb is None or dep is None:
        return None
    m = color_mask(rgb[..., :3], color)
    if m.sum() < 6:
        return None
    pts = cam_points(dep, pk["camera_intrinsics"][cam], pk["camera_poses"][cam], m)
    if len(pts) < 6:
        return None
    # keep points above table, on cube top/side
    pts = pts[(pts[:, 2] > -0.02) & (pts[:, 2] < 0.12)]
    if len(pts) < 6:
        return None
    # top face only (side faces bias the estimate toward the camera)
    zt = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > zt - 0.006]
    if len(top) >= 4:
        pts = top
    med = np.median(pts, 0)
    near = pts[np.linalg.norm(pts[:, :2] - med[:2], axis=1) < 0.04]
    if len(near) < 4:
        near = pts
    return np.median(near, 0), int(m.sum())


# ---------------------------------------------------------------- main
def main(ctx):
    try:
        _main(ctx)
    except Exception as ex:
        import traceback
        log("controller error:", repr(ex), traceback.format_exc()[-1500:])


def _main(ctx):
    t0 = time.time()
    robot = Robot(ctx)
    reply, pk = robot.observe()
    instruction = pk.get("instruction", "")
    log("instruction:", instruction, "step", pk.get("step_id"))
    man = load_manifest(pk)
    if man is None:
        log("no manifest")
        events = []
    else:
        events = parse_events(man["frames"])
    for e in events:
        log("event", e[0], e[1], np.round(e[2], 3).tolist(), np.round(e[3], 3).tolist())
    color = parse_color(instruction)
    vis = locate_cube(reply, pk, color, "front")
    log("vision cube:", None if vis is None else (np.round(vis[0], 3).tolist(), vis[1]))
    chains = build_chains(events)
    for c in chains:
        log("chain start", np.round(c["start"], 3).tolist(), "end", np.round(c["end"], 3).tolist(), "n", len(c["events"]))
    chain = None
    if chains:
        if vis is not None:
            chain = min(chains, key=lambda c: float(np.linalg.norm(c["end"] - vis[0][:2])))
            dmin = float(np.linalg.norm(chain["end"] - vis[0][:2]))
            log("chain matched to vision, dist", round(dmin, 3))
        else:
            chain = max(chains, key=lambda c: len(c["events"]))
            log("no vision; using longest chain")
    all_events = list(events)
    buttons = [e for e in events if e[0] == "button"]
    cev = (chain["events"] if chain else []) + buttons
    cev.sort(key=lambda e: e[1])
    events = cev
    tgt, why = choose_target(events, instruction)
    log("target:", why, None if tgt is None else np.round(tgt[2], 3).tolist())
    if tgt is None:
        log("no target; abort")
        return
    target_xy = tgt[2][:2].copy()
    place_z = float(tgt[2][2])
    # targets swap slots during the video: follow the chosen target's identity
    try:
        import tracker
        vframes = [f for f in man["frames"] if f.get("phase") in ("video_demo", "execution")]
        slots, sev = tracker.swap_analysis(vframes, log=log)
        all_rel = [e for e in all_events if e[0] == "release"]
        offs = []
        for e in all_rel:
            d = np.linalg.norm(slots - e[2][:2], axis=1)
            if d.min() < 0.025:
                offs.append(e[2][:2] - slots[int(d.argmin())])
        off = np.median(np.array(offs), axis=0) if offs else np.zeros(2)
        log("slots", np.round(slots, 3).tolist(), "release-slot offset", np.round(off, 4).tolist(), "n", len(offs))
        d = np.linalg.norm(slots + off - target_xy, axis=1)
        s0 = int(d.argmin())
        log("placement slot", s0, "dist", round(float(d.min()), 3), "at frame", tgt[1])
        if d.min() < 0.04:
            s1 = tracker.target_slot_after(slots, sev, s0, tgt[1])
            target_xy = slots[s1] + off
            log("target identity followed: slot", s0, "->", s1, "xy", np.round(target_xy, 3).tolist())
        else:
            log("placement not at a slot; using raw placement xy")
    except Exception as ex:
        import traceback
        log("tracker failed:", repr(ex), traceback.format_exc()[-500:])

    # cube location: last release in video (cube rests there), refined by vision
    rel = [e for e in events if e[0] == "release"]
    last = rel[-1] if rel else None
    if last is not None:
        cube_xy = last[2][:2].copy()
        grasp_z = float(last[2][2])
        grasp_rv = last[3].copy()
        if vis is not None and np.linalg.norm(vis[0][:2] - cube_xy) < 0.04:
            pass  # expert release xy is the cube centre; vision has a +x bias
        elif vis is not None:
            log("vision disagrees with last release; using vision")
            cube_xy = vis[0][:2]
    else:
        if vis is None:
            log("no cube location")
            return
        cube_xy = vis[0][:2]
        grasp_z = 0.018
        grasp_rv = np.array([math.pi, 0, 0])
    grasp_z = min(max(grasp_z, 0.012), 0.03)
    place_z = min(max(place_z, 0.015), 0.04)
    log("cube_xy", np.round(cube_xy, 3).tolist(), "grasp_z", round(grasp_z, 3))

    hover = 0.12
    # 1. open gripper, go above cube with grasp orientation
    move_to(robot, [cube_xy[0], cube_xy[1], hover], grasp_rv, grip=-1, speed=0.6)
    # refine with wrist? (use front vision again at hover)
    r2, pk2 = robot.observe()
    v2 = locate_cube(r2, pk2, color, "wrist")
    if v2 is not None:
        log("wrist check", np.round(v2[0], 3).tolist())
    move_to(robot, [cube_xy[0], cube_xy[1], grasp_z + 0.03], grasp_rv, grip=-1, speed=0.4)
    move_to(robot, [cube_xy[0], cube_xy[1], grasp_z], grasp_rv, grip=-1, speed=0.2, tol=0.003)
    gripper_cmd(robot, 1, 12)
    grasp_meas_z = float(robot.state[2])
    g = robot.state[6] + robot.state[7]
    log("after close gripper sum", round(float(g), 4), "pos", np.round(robot.state[:3], 3).tolist())
    move_to(robot, [cube_xy[0], cube_xy[1], hover], grasp_rv, grip=1, speed=0.5)
    g = robot.state[6] + robot.state[7]
    log("lifted gripper sum", round(float(g), 4))
    # grasp retry: an empty gripper closes to ~0; re-locate the cube by vision and retry
    for attempt in range(2):
        if robot.ended or g > 0.015:
            break
        log("grasp missed; retry", attempt)
        gripper_cmd(robot, -1, 8)
        rr, pr = robot.observe()
        vv = locate_cube(rr, pr, color, "front")
        if vv is None:
            log("cube not visible for retry")
            break
        cube_xy = vv[0][:2] - np.array([0.008, 0.0])  # front-view top-face +x bias
        move_to(robot, [cube_xy[0], cube_xy[1], hover], grasp_rv, grip=-1, speed=0.5)
        rw, pw = robot.observe()
        vw = locate_cube(rw, pw, color, "wrist")
        if vw is not None and np.linalg.norm(vw[0][:2] - cube_xy) < 0.05:
            cube_xy = vw[0][:2]
        move_to(robot, [cube_xy[0], cube_xy[1], grasp_z + 0.03], grasp_rv, grip=-1, speed=0.4)
        move_to(robot, [cube_xy[0], cube_xy[1], grasp_z], grasp_rv, grip=-1, speed=0.2, tol=0.003)
        gripper_cmd(robot, 1, 12)
        grasp_meas_z = float(robot.state[2])
        move_to(robot, [cube_xy[0], cube_xy[1], hover], grasp_rv, grip=1, speed=0.5)
        g = robot.state[6] + robot.state[7]
        log("retry lifted gripper sum", round(float(g), 4))
    # 2. carry to target
    move_to(robot, [target_xy[0], target_xy[1], hover], tgt[3], grip=1, speed=0.6)
    rel_z = grasp_meas_z + RELEASE_DZ
    log("release z", round(rel_z, 4))
    move_to(robot, [target_xy[0], target_xy[1], rel_z + 0.03], tgt[3], grip=1, speed=0.4)
    move_to(robot, [target_xy[0], target_xy[1], rel_z], tgt[3], grip=1, speed=0.15, tol=0.003)
    robot.step([[0, 0, 0, 0, 0, 0, 1]] * 4)
    log("pre-release pos", np.round(robot.state[:3], 4).tolist())
    gripper_cmd(robot, -1, 10)
    log("post-release ended", robot.ended, "complete", robot.complete)
    move_to(robot, [target_xy[0], target_xy[1], hover], tgt[3], grip=-1, speed=0.5)
    if robot.ended:
        log("episode ended; complete", robot.complete, "steps", robot.step_id, "t", round(time.time() - t0, 1))
        return
    r3, pk3 = robot.observe()
    v3 = locate_cube(r3, pk3, color, "front")
    log("final cube vision", None if v3 is None else np.round(v3[0], 3).tolist(),
        "target", np.round(target_xy, 3).tolist(), "ended", robot.ended, "complete", robot.complete,
        "steps", robot.step_id, "t", round(time.time() - t0, 1))
