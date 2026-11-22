"""RouteStick controller v1: reproduce the drawn path shown in the conditioning video.

The conditioning-video frames in the published manifest carry the robot state
(end-effector pose) of the drawing robot.  We extract that EE trajectory and track it
with closed-loop chunks of official delta rows.  Layout check: compare video frame 0
with the first execution frame (logged; v1 replays absolute coordinates).
"""
import json
import time
from pathlib import Path

import numpy as np

WS = Path("/workspace")
OUT = WS / "output"
STEP_SCALE = 0.05
ROT_SCALE = 0.5
ROT_GAIN = 0.4  # assumed like translation; closed loop corrects residual each chunk
GAIN = 0.4  # measured: one row realises ~40% of the commanded delta (ep1 probe)
CHUNK = 10
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def packet(reply):
    for b in reply.get("content", []):
        if b.get("type") == "text":
            try:
                return json.loads(b["text"])
            except Exception:
                continue
    return {}


def load_video(pk):
    import os
    cands = []
    seq = pk.get("frame_sequence") or {}
    if seq.get("manifest_path"):
        cands.append(WS / seq["manifest_path"])
    if os.environ.get("ROUTESTICK_MANIFEST"):  # development fallback only
        cands.append(Path(os.environ["ROUTESTICK_MANIFEST"]))
    for mp in cands:
        try:
            man = json.loads(Path(mp).read_text())
        except Exception as e:  # noqa
            log("manifest read failed", mp, e)
            continue
        frames = sorted(man["frames"], key=lambda f: f["frame_index"])
        vid = [f for f in frames if f.get("phase") == "video_demo"]
        if vid:
            log("manifest", mp)
            return vid, frames
    return None, None


def img_diff(f_a, f_b):
    try:
        from PIL import Image
        def rgb(f):
            p = [x["path"] for x in f["files"] if x["camera"] == "front" and x["kind"] == "rgb"][0]
            return np.asarray(Image.open(WS / p).convert("RGB"), dtype=np.float32)
        return float(np.abs(rgb(f_a) - rgb(f_b)).mean())
    except Exception as e:  # noqa
        log("img_diff failed", e)
        return None


def step(ctx, rows):
    rows = [[float(np.clip(v, -1, 1)) for v in r] for r in rows]
    pk = packet(ctx.call("robomme_step", actions=rows))
    return pk


def get_orient(pk):
    st = pk.get("state")
    return None if st is None else np.array(st[3:6], dtype=float)


def get_state(pk):
    st = pk.get("state")
    if st is None:
        steps = pk.get("steps") or []
        if steps:
            st = steps[-1].get("state")
    return None if st is None else np.array(st[:3], dtype=float)


def ended(pk):
    if pk.get("episode_ended") or (pk.get("transition") or {}).get("episode_ended"):
        return True
    for s in ((pk.get("transition") or {}).get("steps") or []) + (pk.get("steps") or []):
        if s.get("episode_ended"):
            return True
    return False


def complete(pk):
    tr = pk.get("transition") or {}
    if tr.get("task_complete") or pk.get("task_complete"):
        return True
    for s in ((pk.get("transition") or {}).get("steps") or []) + (pk.get("steps") or []):
        if s.get("task_complete"):
            return True
    return False


def remaining(pk):
    lim = pk.get("episode_limits") or {}
    return lim.get("remaining_episode_steps")


def rv_to_R(r):
    r = np.asarray(r, dtype=float)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_to_rv(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        # near pi: axis from the largest diagonal element
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rot_error(target_rv, cur_rv):
    """World-frame rotation vector e with R_target = exp(e) R_cur."""
    return R_to_rv(rv_to_R(target_rv) @ rv_to_R(cur_rv).T)


def build_orient(vid):
    src = np.array([f.get("source_frame_index", i) for i, f in enumerate(vid)], dtype=float)
    t = np.arange(src[0], src[-1] + 1)
    idx = np.searchsorted(src, t).clip(0, len(vid) - 1)
    return np.array([vid[i]["state"][3:6] for i in idx], dtype=float)


def build_path(vid):
    """Waypoints at the video's source clock (one per source frame, interpolated)."""
    src = np.array([f.get("source_frame_index", i) for i, f in enumerate(vid)], dtype=float)
    P = np.array([f["state"][:3] for f in vid], dtype=float)
    t = np.arange(src[0], src[-1] + 1)
    W = np.stack([np.interp(t, src, P[:, k]) for k in range(3)], 1)
    return W


def track(ctx, W, pk, speed=1.0, max_steps=None, O=None):
    """Follow waypoint array W (N,3) with closed-loop chunks; O (N,3) target orientations."""
    cur = get_state(pk)
    n = len(W)
    # resample by speed factor
    if speed != 1.0:
        idx = np.arange(0, n - 1 + 1e-9, speed)
        W = np.stack([np.interp(idx, np.arange(n), W[:, k]) for k in range(3)], 1)
        if O is not None:
            O = O[np.round(idx).astype(int).clip(0, len(O) - 1)]
        n = len(W)
    max_rot = 0.0
    i = 0
    used = 0
    max_err = 0.0
    while i < n:
        rem = remaining(pk)
        if rem is not None and rem <= 2:
            log("out of steps")
            break
        k = min(CHUNK, n - i)
        if rem is not None:
            k = min(k, rem - 1)
        rows = []
        prev = cur.copy()
        e = np.zeros(3)
        if O is not None:
            e = rot_error(O[min(i + k - 1, n - 1)], get_orient(pk))
            max_rot = max(max_rot, float(np.linalg.norm(e)))
        for j in range(k):
            tgt = W[i + j]
            d = (tgt - prev) / (STEP_SCALE * GAIN)
            d = np.clip(d, -1, 1)
            r = np.clip(e / (ROT_SCALE * ROT_GAIN), -0.3, 0.3)
            e = e - r * ROT_SCALE * ROT_GAIN
            rows.append([d[0], d[1], d[2], r[0], r[1], r[2], 0])
            prev = prev + d * STEP_SCALE * GAIN
        pk = step(ctx, rows)
        used += k
        if ended(pk):
            log("episode ended during track at", pk.get("step_id"), "task_complete", complete(pk))
            return pk, True
        cur = get_state(pk)
        err = float(np.linalg.norm(cur[:2] - W[i + k - 1][:2]))
        max_err = max(max_err, err)
        i += k
    log(f"track done steps={used} max_xy_err={max_err:.4f} max_rot_err={max_rot:.4f} final={cur} orient={get_orient(pk)}")
    return pk, False


def map_to_execution(W, f_video, f_exec):
    """Warp the video's path onto the execution layout using detected target discs."""
    try:
        import perception as P
        Dv, _, _ = P.detect_discs(f_video)
        De, _, _ = P.detect_discs(f_exec)
        log("discs video", np.round(Dv, 3).tolist())
        log("discs exec ", np.round(De, 3).tolist())
        if len(Dv) == len(De) and len(Dv) >= 2:
            wp = P.RowWarp(Dv, De)
            W2 = P.warp_path(W, wp)
            log("row warp applied; max shift", round(float(np.abs(W2 - W).max()), 4))
            return W2
        if len(Dv) >= 2 and len(De) >= 2:
            Am, Bm = P.match_sets(Dv, De)
            if len(Am) >= 2:
                R, t = P.fit_rigid(Am, Bm)
                W2 = W.copy()
                W2[:, :2] = W[:, :2] @ R.T + t
                log("rigid fallback applied", np.round(t, 4).tolist())
                return W2
        log("no warp (disc detection mismatch)")
    except Exception as e:  # noqa
        import traceback
        log("warp failed", repr(e), traceback.format_exc())
    return W


def main(ctx):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        _main(ctx, t0)
    except Exception as e:  # never crash silently; log and return
        import traceback
        log("ERROR", repr(e), traceback.format_exc())
    finally:
        log("wall", round(time.time() - t0, 1))
        try:
            (OUT / "log.txt").write_text("\n".join(LOG))
        except Exception:
            pass


def _main(ctx, t0):
    pk = packet(ctx.call("robomme_observe"))
    log("instruction:", pk.get("instruction"), "step", pk.get("step_id"))
    vid, frames = load_video(pk)
    if not vid:
        log("no video frames; abort")
        return
    log("video frames", len(vid))
    log("front diff video0 vs exec:", img_diff(vid[0], frames[-1]))
    W = build_path(vid)
    W = map_to_execution(W, vid[0], frames[-1])
    np.save(OUT / "path.npy", W)
    zmin = W[:, 2].min()
    k0 = int(np.argmax(W[:, 2] < zmin + 0.004))  # first index at drawing height
    log("path len", len(W), "draw start idx", k0, "zmin", round(float(zmin), 4))
    O = build_orient(vid)
    pk, done = track(ctx, W, pk, O=O)
    if not done:
        pk = step(ctx, [[0, 0, 0, 0, 0, 0, 0]] * 3)
        done = ended(pk)
    log("pass1 ended", done, "complete", complete(pk), "step", pk.get("step_id"))
    log("final step", pk.get("step_id"), "ended", ended(pk), "task_complete", complete(pk))
