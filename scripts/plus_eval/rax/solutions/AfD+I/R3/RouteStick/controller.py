"""RouteStick controller v1: re-trace the drawn path recorded in the current episode's
conditioning video.

The reset frame sequence (published by robomme_observe at step 0) carries the robot
`state` of every conditioning frame; the video robot drew the path with its stick, so
its TCP trajectory *is* the path to reproduce.  We verify the execution scene matches
the first video frame, then track the trajectory closed-loop in chunks.
"""
import json
import os
import time
from pathlib import Path

import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WS = Path("/workspace")
OUT = WS / "output"
STEP_SCALE = 0.05      # m per unit action
MAX_UNIT = 0.9         # max |action| per axis per row
CHUNK = 12             # waypoints per robomme_step call
LOG = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(f"{time.time():.2f} {msg}\n")
    except Exception:
        pass


def packet(reply):
    for block in reply.get("content", []):
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except Exception:
                continue
    return {}


def rotvec_to_R(r):
    r = np.asarray(r, float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def R_to_rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: robust axis extraction
        M = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(M)))
        k = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_correction(cur_rv, tgt_rv, gain=0.5, max_unit=0.2):
    """World-frame axis-angle delta (in action units) moving cur toward tgt."""
    Rc, Rt = rotvec_to_R(cur_rv), rotvec_to_R(tgt_rv)
    err = R_to_rotvec(Rt @ Rc.T)  # world frame
    u = gain * err / 0.5
    return np.clip(u, -max_unit, max_unit)


def load_video_path(obs):
    seq = obs.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    if not mp:
        raise RuntimeError("no frame_sequence manifest in observation")
    man = json.loads((WS / mp).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f.get("phase") == "video_demo" and f.get("state")]
    exe = [f for f in frames if f.get("phase") == "execution"]
    return man, vid, exe


def scene_check(vid, exe):
    """Mean abs pixel difference between first video frame and execution frame (front)."""
    try:
        from PIL import Image

        def front(fr):
            for fl in fr["files"]:
                if fl["camera"] == "front" and fl["kind"] == "rgb":
                    return np.asarray(Image.open(WS / fl["path"]).convert("RGB")).astype(int)
        a, b = front(vid[0]), front(exe[-1])
        d = np.abs(a - b).sum(2)
        return float(d.mean()), int((d > 40).sum())
    except Exception as e:
        log("scene_check failed", repr(e))
        return None, None


def _front_rgbd(fr):
    from PIL import Image
    p = {(x["camera"], x["kind"]): x["path"] for x in fr["files"]}
    rgb = np.asarray(Image.open(WS / p["front", "rgb"]).convert("RGB"))
    dep = np.asarray(Image.open(WS / p["front", "depth"])).astype(float) * 0.001
    return rgb, dep, fr["camera_intrinsics"]["front"]["matrix"], fr["camera_poses"]["front"]


def remap_to_execution(path, vfr, efr):
    """Map the video path from the video layout onto the execution layout via the discs."""
    from discs import detect_discs, filter_discs, find_anchors, map_path
    vd = filter_discs(detect_discs(*_front_rgbd(vfr)))
    ed = filter_discs(detect_discs(*_front_rgbd(efr)))
    log("video discs", [np.round(d["xy"], 3).tolist() for d in vd])
    log("exec discs ", [np.round(d["xy"], 3).tolist() for d in ed])
    if len(vd) < 2 or len(ed) < 2:
        log("too few discs; no remap")
        return path
    anchors = find_anchors(path, vd)
    log("anchors (path idx, disc)", anchors)
    if not anchors:
        return path
    V = np.array([d["xy"] for d in vd])
    E = np.array([d["xy"] for d in ed])
    if len(vd) == len(ed):
        Emap = list(E)
    else:
        # unequal counts: match each video disc to the nearest exec disc after centroid alignment
        Ec = E - E.mean(0) + V.mean(0)
        Emap = [E[int(np.argmin(np.linalg.norm(Ec - v, axis=1)))] for v in V]
        log("unequal disc counts", len(vd), len(ed), "nearest-match used")
    shift = np.linalg.norm(np.array(Emap) - V, axis=1)
    log("disc shifts", np.round(shift, 4).tolist())
    if shift.max() < 0.003:
        log("layout unchanged; using video path as is")
        return path
    mp = map_path(path, vd, Emap, anchors)
    return mp


def make_rows(start, targets, rot_u=None):
    """Rows of position deltas walking start->targets[0]->targets[1]..., splitting long hops."""
    rows = []
    prev = np.asarray(start, float)
    for i, t in enumerate(targets):
        d = (np.asarray(t, float) - prev) / STEP_SCALE
        n = max(1, int(np.ceil(np.max(np.abs(d)) / MAX_UNIT - 1e-9)))
        for k in range(n):
            r = [float(x) for x in d / n] + [0.0, 0.0, 0.0, 0.0]
            rows.append(r)
        prev = np.asarray(t, float)
    if rot_u is not None and rows:
        rows[0][3:6] = [float(x) for x in rot_u]
    return rows


class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = None
        self.ended = False
        self.complete = False
        self.remaining = None
        self.step_id = 0

    def update(self, p):
        if p.get("state"):
            self.state = np.asarray(p["state"], float)
        lim = p.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        self.step_id = p.get("step_id", self.step_id)
        if p.get("episode_ended"):
            self.ended = True
        for s in (p.get("transition") or {}).get("steps", []) or []:
            if s.get("task_complete"):
                self.complete = True
            if s.get("episode_ended"):
                self.ended = True
        if p.get("task_complete"):
            self.complete = True

    def observe(self):
        p = packet(self.ctx.call("robomme_observe"))
        self.update(p)
        return p

    def step(self, rows):
        if self.ended or not rows:
            return {}
        if self.remaining is not None:
            rows = rows[: max(0, self.remaining)]
            if not rows:
                return {}
        out = {}
        for i in range(0, len(rows), 100):
            out = packet(self.ctx.call("robomme_step", actions=rows[i:i + 100]))
            self.update(out)
            if self.ended:
                break
        return out


A_NOW = 0.26     # fraction of commanded delta realised in the same step
A_PREV = 0.14    # carry-over from the previous step's command
SPEED_LOW = 0.004    # m/step along the drawing part of the path
SPEED_HIGH = 0.012   # m/step while the stick is well above the table
LOW_Z_MARGIN = 0.03  # "drawing" if z < min_z + margin
DWELL_HOLD = 4       # extra steps held where the video robot paused (disc contacts)


def resample(path, low_z):
    """Arc-length resample: slower near the table, keep short holds at video dwells."""
    out = [path[0]]
    seg_d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    acc = 0.0
    dwell = 0
    for i in range(1, len(path)):
        a, b, d = path[i - 1], path[i], seg_d[i - 1]
        if d < 2e-4:
            dwell += 1
            if dwell == 3:
                # the video robot paused here (disc contact): go exactly there and hold
                out.extend([a.copy()] * DWELL_HOLD)
                acc = 0.0
            continue
        dwell = 0
        v = SPEED_LOW if min(a[2], b[2]) < low_z else SPEED_HIGH
        # walk along segment a->b emitting points every v metres of travel
        s = v - acc
        while s <= d + 1e-12:
            out.append(a + (b - a) * (s / d))
            s += v
        acc = d - (s - v)
    if np.linalg.norm(out[-1] - path[-1]) > 1e-6:
        out.append(path[-1])
    return np.array(out)


def track(rob, ref, t0, wall_limit=1050.0):
    """Per-step closed loop: invert dx = 0.05*(A_NOW*u_k + A_PREV*u_{k-1})."""
    u_prev = np.zeros(3)
    errs = []
    for k, r in enumerate(ref):
        if rob.ended or (rob.remaining is not None and rob.remaining <= 0):
            break
        x = rob.state[:3]
        errs.append(float(np.linalg.norm(x - r)))
        u = (r - x - STEP_SCALE * A_PREV * u_prev) / (STEP_SCALE * A_NOW)
        u = np.clip(u, -1.0, 1.0)
        rob.step([[float(u[0]), float(u[1]), float(u[2]), 0.0, 0.0, 0.0, 0.0]])
        u_prev = u
        if k % 50 == 0:
            log(f"ref {k}/{len(ref)} step {rob.step_id} err {errs[-1]:.4f}")
        if time.time() - t0 > wall_limit:
            log("wall-time guard hit")
            break
    return errs, u_prev


def main(ctx, start_index=0):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rob = Robot(ctx)
    obs = rob.observe()
    log("instruction:", obs.get("instruction"), "step", obs.get("step_id"), "limits", obs.get("episode_limits"))
    man, vid, exe = load_video_path(obs)
    log("video frames with state:", len(vid), "exec frames:", len(exe))
    if exe:
        log("scene diff (mean, n>40):", scene_check(vid, exe))
    path = np.array([f["state"][:3] for f in vid], float)[start_index:]
    np.save(OUT / "video_path.npy", path)
    if exe and start_index == 0:
        try:
            path = remap_to_execution(path, vid[0], exe[-1])
        except Exception as e:
            log("remap failed, using video path unchanged:", repr(e))
    np.save(OUT / "mapped_path.npy", path)
    low_z = float(path[:, 2].min()) + LOW_Z_MARGIN
    log("path z range", path[:, 2].min(), path[:, 2].max(), "n", len(path))
    ref = resample(path, low_z)
    # time budget: if the reference would not fit the remaining steps, speed it up
    rem = rob.remaining if rob.remaining is not None else 1190
    budget = rem - 40
    if len(ref) > budget:
        idx = np.linspace(0, len(ref) - 1, budget).round().astype(int)
        ref = ref[idx]
        log("reference compressed to", len(ref))
    np.save(OUT / "reference.npy", ref)
    log("reference steps", len(ref), "remaining", rem)
    errs, u_prev = track(rob, ref, t0)
    if errs:
        log("tracking done: step", rob.step_id, "ended", rob.ended, "complete", rob.complete,
            "err mean", float(np.mean(errs)), "p95", float(np.percentile(errs, 95)), "max", float(np.max(errs)))
    # settle on the final point, then hold so the simulator can register it
    if not rob.ended:
        track(rob, [ref[-1]] * 6, t0)
        rob.step([[0.0] * 7 for _ in range(10)])
        log("after hold: ended", rob.ended, "complete", rob.complete, "step", rob.step_id)
    if not rob.ended:
        rob.observe()
    log("final: ended", rob.ended, "complete", rob.complete, "step", rob.step_id, "wall", time.time() - t0)
    return {"ended": rob.ended, "complete": rob.complete}
