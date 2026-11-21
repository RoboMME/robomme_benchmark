"""PatternLock controller (RAX, interaction-only). v2: grid-index mapping video->execution (scenes differ).

Pipeline (one log line per decision to stdout and output/patternlock_log.txt):
  P0 PERCEIVE  observe step 0 -> conditioning-video manifest. Fit a (rotated) button lattice on
               video frame 0 depth AND on the execution depth separately (perception.fit_grid).
               Pattern = video nodes where the expert's TCP dwelt (state; primary), cross-checked with
               the order buttons turn red in the video front view and RDP path corners.
               Map pattern by grid index (i ~ world x, j ~ world y) onto the execution lattice.
               exits: ok -> P1; no plan -> hold 2 steps and return.
  P1 APPROACH  hover above first corner (+5 cm), descend to touch height (video touch z).
  P2 TRACE     straight segments between corners at touch height; segments that would cross an
               intermediate node use a parallel detour between lattice lines. Dwell 9 steps at each
               corner, check it is red in front view; if not: dwell 6 more, nudge down 3 mm, dwell 6
               (only when step budget allows).
  P3 HOLD      5 idle steps. Return.
Tool errors are caught and logged; after an error the robot is re-observed, never blindly retried.
"""
import base64
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geom import rotvec_to_R, R_to_rotvec  # noqa: E402
import perception as Pc  # noqa: E402

VERSION = "v2"
WS = Path("/workspace")
OUT = WS / "output"

# ---- tunables (tested in ep1 unless noted) ----
DEFAULT_OFFSET = np.array([-0.006, -0.002])  # TCP - button xy at touch (ep1 video)
DEFAULT_TOUCH_Z = 0.070                 # TCP z while touching (ep1 video)
HOVER_DZ = 0.05                         # hover above touch height
VMAX = 0.30                             # action units per step (0.015 m/step)

LOG_LINES = []


def log(msg):
    line = f"[{VERSION} t={time.time():.1f}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "patternlock_log.txt", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def meta_of(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                return {}
    return {}


def image_of(reply, meta, camera, kind="rgb"):
    for a in meta.get("attachments", []):
        if a.get("camera") == camera and a.get("kind") == kind:
            blk = reply["content"][a["content_index"]]
            data = base64.b64decode(blk["data"])
            im = Image.open(io.BytesIO(data))
            return np.asarray(im)
    return None


# ------------------------- helpers -------------------------

def find_initial_manifest(episode_id):
    """Fallback when the observe reply lacks frame_sequence: newest sequence_000000 of this episode."""
    best = None
    for p in (WS / "runtime/autonomous_controller/runs").glob("*/frames/sequence_000000/manifest.json"):
        try:
            man = json.loads(p.read_text())
            if man.get("transition", {}).get("episode_id") != episode_id:
                continue
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
        except Exception:
            continue
    return str(best.relative_to(WS)) if best else None


def check_red(rob, npix):
    try:
        rgb = image_of(rob.reply, rob.meta, "front", "rgb")
        if rgb is None:
            return set()
        return Pc.red_nodes(rgb[..., :3], npix)
    except Exception:
        log("check_red exception\n" + traceback.format_exc())
        return set()


# ------------------------- control -------------------------

class Robot:
    def __init__(self, ctx):
        self.ctx = ctx
        self.meta = None
        self.reply = None
        self.ended = False
        self.complete = False
        self.remaining = 10 ** 6
        self.R0 = None

    def _update(self, reply):
        self.reply = reply
        self.meta = meta_of(reply)
        m = self.meta
        if m.get("episode_ended") or m.get("transition", {}).get("episode_ended"):
            self.ended = True
        if m.get("task_complete") or m.get("transition", {}).get("task_complete"):
            self.complete = True
        lim = m.get("episode_limits") or {}
        if "remaining_episode_steps" in lim:
            self.remaining = lim["remaining_episode_steps"]
        for st in m.get("transition", {}).get("steps", []) or []:
            if st.get("task_complete"):
                self.complete = True
            fb = st.get("feedback")
            if fb:
                log(f"feedback step {st.get('step_id')}: {fb}")

    def observe(self):
        self._update(self.ctx.call("robomme_observe"))
        return self.meta

    @property
    def state(self):
        return np.asarray(self.meta["state"], dtype=float)

    def step(self, rows):
        if self.ended:
            return False
        rows = rows[: max(1, min(100, self.remaining))]
        try:
            self._update(self.ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows]))
        except Exception as e:
            log(f"step error: {e!r}; re-observing")
            try:
                self.observe()
            except Exception as e2:
                log(f"observe error: {e2!r}")
            return False
        return True

    def rot_row(self):
        if self.R0 is None:
            return np.zeros(3)
        Rc = rotvec_to_R(self.state[3:6])
        err = R_to_rotvec(self.R0 @ Rc.T)
        return np.clip(err / 0.5, -0.2, 0.2)

    def move_to(self, target, tol=0.004, vmax=VMAX, max_steps=120, chunk=1):
        target = np.asarray(target, dtype=float)
        n = 0
        stall = 0
        last = None
        while n < max_steps and not self.ended:
            pos = self.state[:3]
            err = target - pos
            dist = float(np.linalg.norm(err))
            if dist < tol:
                return True, dist
            if last is not None and abs(last - dist) < 2e-4:
                stall += 1
                if stall > 8:
                    log(f"move_to stall at dist {dist:.4f}")
                    return False, dist
            else:
                stall = 0
            last = dist
            a = err / 0.05
            norm = float(np.linalg.norm(a))
            if norm > vmax:
                a = a * (vmax / norm)
            row = list(a) + list(self.rot_row()) + [0.0]
            self.step([row] * chunk)
            n += chunk
        return float(np.linalg.norm(target - self.state[:3])) < tol, float(np.linalg.norm(target - self.state[:3]))

    def hold(self, k):
        for _ in range(k):
            if self.ended:
                return
            self.step([[0, 0, 0] + list(self.rot_row()) + [0.0]])


# ======================= v2 =======================

SEG_VMAX = 0.30          # ~0.006 m/step actual (expert video ~0.006 m/frame)
CORNER_TOL = 0.003
VIA_TOL = 0.008
DWELL_CORNER = 9         # red appeared ~7 frames after arrival in both videos
EXTRA_DWELL = 6
NUDGE_DZ = 0.003


def frame_files(fr):
    return {(f["camera"], f["kind"]): f for f in fr.get("files", [])}


def grid_from_manifest_frame(fr):
    files = frame_files(fr)
    blobs = {}
    for cam in ("wrist", "front"):
        f = files.get((cam, "depth"))
        if not f:
            blobs[cam] = []
            continue
        dep = np.asarray(Image.open(WS / f["path"])).astype(float) * f.get("meters_per_unit", 0.001)
        b, _ = Pc.detect_blobs(dep, fr["camera_intrinsics"][cam]["matrix"], fr["camera_poses"][cam])
        blobs[cam] = b
    pts = Pc.merge_views(blobs["wrist"], blobs["front"])
    g = Pc.fit_grid(pts)
    top = [b["z"] for b in blobs["wrist"] + blobs["front"]]
    return g, (float(np.median(top)) if top else 0.016), {k: len(v) for k, v in blobs.items()}


def grid_from_reply(reply, meta):
    blobs = {}
    for cam in ("wrist", "front"):
        dep = image_of(reply, meta, cam, "depth")
        if dep is None:
            blobs[cam] = []
            continue
        mpu = (meta.get("depth", {}).get(cam, {}) or {}).get("meters_per_unit", 0.001)
        b, _ = Pc.detect_blobs(dep.astype(float) * mpu, meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam])
        blobs[cam] = b
    pts = Pc.merge_views(blobs["wrist"], blobs["front"])
    g = Pc.fit_grid(pts)
    top = [b["z"] for b in blobs["wrist"] + blobs["front"]]
    return g, (float(np.median(top)) if top else 0.016), {k: len(v) for k, v in blobs.items()}


def gstr(g):
    if not g:
        return "None"
    return (f"theta={np.degrees(g['theta']):.2f}deg s={g['spacing']:.4f} shape={g['ni']}x{g['nj']} n={g['n']} "
            f"resid_med={g['resid_med']:.4f} max={g['resid_max']:.4f} origin={np.round(g['A'][2], 4).tolist()}")


def perceive2(rob, m):
    seqd = m.get("frame_sequence") or {}
    mpath = seqd.get("manifest_path") or find_initial_manifest(m.get("episode_id"))
    frames = []
    if mpath:
        frames = json.loads((WS / mpath).read_text()).get("frames", [])
    video = [f for f in frames if f.get("phase") == "video_demo"]
    log(f"manifest={mpath} frames={len(frames)} video={len(video)}")
    if not video:
        return None
    # video grid from first video frame (fallback: a few later frames)
    gv = None
    for fr in video[:1] + video[1:40:13]:
        gv, vtop, cnt = grid_from_manifest_frame(fr)
        log(f"video grid (frame {fr.get('frame_index')}) blobs={cnt} {gstr(gv)}")
        if gv and gv["n"] >= 9:
            break
    ge, etop, cnt = grid_from_reply(rob.reply, m)
    log(f"exec grid blobs={cnt} {gstr(ge)}")
    if not gv or not ge:
        return None
    if (gv["ni"], gv["nj"]) != (ge["ni"], ge["nj"]):
        log("WARNING grid shapes differ")
    states = [f.get("state") for f in video if f.get("state") is not None]
    rs = Pc.sequence_from_states(states, gv, DEFAULT_OFFSET) if len(states) == len(video) else None
    corners_d = rs["dwell_nodes"] if rs else []
    corners_k = rs["corner_nodes"] if rs else []
    if rs:
        log(f"state: dwell={corners_d} rdp={corners_k} visited={rs['seq']} n_slow={rs['n_slow']} offset={np.round(rs['offset'], 4).tolist()} touch_z={rs['touch_z']:.4f} visit_d={rs['visit_d']}")
    corners_r = []
    try:
        fr0 = video[0]
        npix_v = Pc.node_pixels(Pc.all_nodes(gv), vtop, fr0["camera_intrinsics"]["front"]["matrix"], fr0["camera_poses"]["front"])
        rgbs = []
        for fr in video:
            f = frame_files(fr).get(("front", "rgb"))
            if f:
                rgbs.append(np.asarray(Image.open(WS / f["path"]).convert("RGB")))
        corners_r = Pc.sequence_from_red(rgbs, npix_v)
    except Exception:
        log("red seq exception\n" + traceback.format_exc())
    log(f"red: {corners_r} agree_dwell={corners_r == corners_d} agree_rdp={corners_r == corners_k}")
    # primary: expert dwell points (state); then red order; then RDP corners
    if corners_d:
        corners = corners_d
        if corners_r and corners_r != corners_d:
            log("WARNING dwell/red disagree; using dwell")
    elif corners_r:
        corners = corners_r
    else:
        corners = corners_k
    if not corners:
        return None
    # clip to exec grid
    corners = [c for c in corners if 0 <= c[0] < ge["ni"] and 0 <= c[1] < ge["nj"]]
    offset = rs["offset"] if rs else DEFAULT_OFFSET.copy()
    touch_z = rs["touch_z"] if rs else DEFAULT_TOUCH_Z
    if not (DEFAULT_TOUCH_Z - 0.02 < touch_z < DEFAULT_TOUCH_Z + 0.02):
        touch_z = DEFAULT_TOUCH_Z
    touch_z += (etop - vtop) if abs(etop - vtop) < 0.02 else 0.0
    wps = Pc.plan_path(corners, ge)
    nodes_e = Pc.all_nodes(ge)
    clr = Pc.path_min_clearance(wps, nodes_e, corners)
    npix_e = Pc.node_pixels(nodes_e, etop, m["camera_intrinsics"]["front"]["matrix"], m["camera_poses"]["front"])
    log(f"plan corners={corners} n_wp={len(wps)} clearance={clr:.4f} ({clr / ge['spacing']:.2f}s) offset={np.round(offset, 4).tolist()} touch_z={touch_z:.4f}")
    return {"corners": corners, "wps": wps, "offset": offset, "touch_z": touch_z, "npix": npix_e}


def main(ctx):
    t0 = time.time()
    log("start")
    rob = Robot(ctx)
    try:
        m = rob.observe()
    except Exception as e:
        log(f"initial observe failed {e!r}")
        return
    rob.R0 = rotvec_to_R(rob.state[3:6])
    log(f"instruction={m.get('instruction')!r} state={np.round(rob.state, 4).tolist()} limits={m.get('episode_limits')}")
    plan = None
    try:
        plan = perceive2(rob, m)
    except Exception:
        log("perceive exception\n" + traceback.format_exc())
    if not plan:
        log("no plan; holding")
        rob.hold(2)
        return
    off, tz = plan["offset"], plan["touch_z"]

    def T(xy, z):
        return np.array([xy[0] + off[0], xy[1] + off[1], z])

    wps = plan["wps"]
    log(f"P0 done wall={time.time() - t0:.1f}s")
    # ---------- P1 APPROACH ----------
    try:
        ok, d = rob.move_to(T(wps[0][0], tz + HOVER_DZ), tol=0.006, max_steps=150)
        log(f"P1 hover ok={ok} d={d:.4f} step={rob.meta.get('step_id')}")
        ok, d = rob.move_to(T(wps[0][0], tz), tol=CORNER_TOL, vmax=0.15, max_steps=60)
        log(f"P1 descend ok={ok} d={d:.4f} pos={np.round(rob.state[:3], 4).tolist()} step={rob.meta.get('step_id')}")
    except Exception:
        log("P1 exception\n" + traceback.format_exc())
    # ---------- P2 TRACE ----------
    lit = []
    n_corners_left = len(plan["corners"])
    for k, (xy, kind, node) in enumerate(wps):
        if rob.ended:
            break
        try:
            if k > 0:
                tol = CORNER_TOL if kind == "corner" else VIA_TOL
                ok, d = rob.move_to(T(xy, tz), tol=tol, vmax=SEG_VMAX, max_steps=150)
                log(f"P2 wp {k} {kind} {node} ok={ok} d={d:.4f} pos={np.round(rob.state[:3], 4).tolist()} step={rob.meta.get('step_id')} rem={rob.remaining}")
            if kind != "corner":
                continue
            n_corners_left -= 1
            rob.hold(DWELL_CORNER)
            red = check_red(rob, plan["npix"])
            got = node in red
            log(f"P2 corner {node} dwell red={sorted(red)} lit={got} ended={rob.ended} complete={rob.complete}")
            if not got and not rob.ended and rob.remaining > 40 + 30 * n_corners_left:
                rob.hold(EXTRA_DWELL)
                red = check_red(rob, plan["npix"])
                got = node in red
                if not got and not rob.ended:
                    rob.move_to(T(xy, tz - NUDGE_DZ), tol=0.002, vmax=0.1, max_steps=15)
                    rob.hold(EXTRA_DWELL)
                    red = check_red(rob, plan["npix"])
                    got = node in red
                    rob.move_to(T(xy, tz), tol=0.002, vmax=0.1, max_steps=10)
                log(f"P2 corner {node} retry red={sorted(red)} lit={got} ended={rob.ended}")
            lit.append(got)
        except Exception:
            log("P2 exception\n" + traceback.format_exc())
    log(f"P2 done lit={lit} ended={rob.ended} complete={rob.complete} step={rob.meta.get('step_id')}")
    # ---------- P3 HOLD ----------
    try:
        if not rob.ended:
            rob.hold(5)
        log(f"P3 end step={rob.meta.get('step_id')} ended={rob.ended} complete={rob.complete} remaining={rob.remaining} wall={time.time() - t0:.1f}s")
    except Exception:
        log("P3 exception\n" + traceback.format_exc())
