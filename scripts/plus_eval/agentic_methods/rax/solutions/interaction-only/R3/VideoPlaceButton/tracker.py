"""Track target identities through the conditioning video (targets swap positions)."""
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
from geom import Cam
import perception as P

GATE = 0.11       # max association distance (m) from prediction
MIN_N = 12        # min pink points for a detection
TARGET_COLOR = "pink"   # switched to "white" by the controller when no pink targets are visible
EXCLUDE_XY = []         # world xy points (e.g. the button) whose clusters are not targets


def detect_frame(f, root="/workspace"):
    files = {(x["camera"], x["kind"]): x for x in f["files"]}
    rgb = np.array(Image.open(Path(root) / files[("front", "rgb")]["path"]).convert("RGB"))
    dimg = files[("front", "depth")]
    dep = np.array(Image.open(Path(root) / dimg["path"])).astype(float) * float(dimg.get("meters_per_unit", 0.001))
    cam = Cam(f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    tg = P.find_targets(rgb, dep, cam, TARGET_COLOR)
    tg = [t for t in tg if not any(np.linalg.norm(t["xy"] - np.asarray(e)) < 0.06 for e in EXCLUDE_XY)]
    return [t for t in tg if t["n"] >= MIN_N]


def _assign(pred, dets):
    """Min-cost assignment tracks->detections (brute force, small K). Returns dict track->det."""
    K, D = len(pred), len(dets)
    if D == 0:
        return {}
    C = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
    best, best_cost = {}, None
    # each track picks a distinct detection or None
    opts = list(range(D)) + [None] * K
    seen = set()
    for perm in itertools.permutations(range(D + K), K):
        key = tuple(p if p < D else None for p in perm)
        if key in seen:
            continue
        seen.add(key)
        cost = 0.0
        ok = True
        for k, p in enumerate(key):
            if p is None:
                cost += GATE
            else:
                if C[k, p] > GATE:
                    ok = False; break
                cost += C[k, p]
        if ok and (best_cost is None or cost < best_cost):
            best_cost, best = cost, {k: p for k, p in enumerate(key) if p is not None}
    return best


def track(frames, root="/workspace", log=None):
    """frames: ordered video_demo frames (+ optionally the execution frame).
    Returns (positions[T,K,2], frame_indices[T], dets_per_frame)."""
    dets_all = [detect_frame(f, root) for f in frames]
    counts = [len(d) for d in dets_all]
    K = int(np.bincount(counts).argmax()) if counts else 0
    # init at first frame with K detections and all fairly large
    t0 = next((i for i, d in enumerate(dets_all) if len(d) == K), 0)
    pos = np.array([d["xy"] for d in dets_all[t0]], float)
    vel = np.zeros_like(pos)
    T = len(frames)
    out = np.zeros((T, K, 2))
    out[: t0 + 1] = pos
    miss = np.zeros(K, int)
    for t in range(t0 + 1, T):
        dets = np.array([d["xy"] for d in dets_all[t]], float).reshape(-1, 2)
        pred = pos + vel
        a = _assign(pred, dets) if len(dets) else {}
        # a big cluster can be two merged targets: allow two tracks to share one detection when
        # the count dropped and an unmatched track's prediction is near the shared one
        newpos = pred.copy()
        for k, p in a.items():
            newpos[k] = dets[p]
        for k in range(K):
            if k in a:
                v = newpos[k] - pos[k]
                vel[k] = 0.6 * v + 0.4 * vel[k] if miss[k] == 0 else v
                miss[k] = 0
            else:
                miss[k] += 1
                vel[k] *= 0.7
        pos = newpos
        out[t] = pos
    return out, K, dets_all, t0
