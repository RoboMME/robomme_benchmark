"""Conditioning-video analysis: find the picked object and track it to the end."""
import json
import numpy as np
from perception import WS, count_objects, frame_object_xy, assign_step, track, track2, track3, frame_blobs_fused, frame_points, \
    button_location, estimate_table_z, set_exclusions

CLOSED_SUM = 0.07   # finger sum below this = gripper closed (open = 0.08)


def find_release(frames):
    vid = [f for f in frames if f["phase"] == "video_demo"]
    op = np.array([f["state"][6] + f["state"][7] for f in vid])
    closed = np.nonzero(op < CLOSED_SUM)[0]
    if len(closed) == 0:
        return None
    last_closed = closed[-1]
    first_closed = closed[0]
    k_rel = min(last_closed + 1, len(vid) - 1)
    # release xy: EE xy at the last closed frame (object placed under gripper)
    rel_xy = np.array(vid[last_closed]["state"][:2])
    grasp_xy = np.array(vid[first_closed]["state"][:2])
    # tracking start: first frame after release with EE well above the table
    start = None
    for k in range(k_rel, len(vid)):
        if vid[k]["state"][2] > 0.15:
            start = k
            break
    if start is None:
        start = k_rel
    return {"first_closed": int(first_closed), "last_closed": int(last_closed), "release_xy": rel_xy,
            "grasp_xy": grasp_xy, "start": int(start)}


def kmeans_init(P, C0, iters=10):
    C = np.array(C0, float)
    for _ in range(iters):
        d = np.linalg.norm(P[:, None] - C[None], axis=2)
        a = d.argmin(1)
        for j in range(len(C)):
            if (a == j).sum() > 0:
                C[j] = P[a == j].mean(0)
    return C


def analyze(manifest_path, log=print, ws=WS):
    man = json.loads((ws / manifest_path).read_text()) if not str(manifest_path).startswith("/") else json.loads(open(manifest_path).read())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    ev = find_release(frames)
    if ev is None:
        log("video: no grasp found")
        return None
    log(f"video: grasp frames {ev['first_closed']}..{ev['last_closed']} grasp_xy={np.round(ev['grasp_xy'],3)} "
        f"release_xy={np.round(ev['release_xy'],3)} start={ev['start']}")
    rgb0, pts0 = frame_points(frames[0], "front", ws)
    btn = button_location(rgb0, pts0, estimate_table_z(pts0, rgb0))
    if btn is not None:
        set_exclusions([btn["xy"]])
        log(f"video: button at {np.round(btn['xy'],3)} excluded from object mask")
    tz = estimate_table_z(pts0, rgb0)
    # object count from the first frame (before interaction), fused front+wrist grid
    B0 = frame_blobs_fused(frames[0], ws, tz)
    N = len(B0)
    unit = float(np.median([len(b) for b in B0])) if B0 else 40.0
    vid_end = max(i for i, f in enumerate(frames) if f["phase"] == "video_demo")
    start = ev["start"]
    C = None
    for k in range(ev["start"], min(ev["start"] + 60, vid_end)):
        if frames[k]["state"][2] < 0.15:
            continue
        Bk = frame_blobs_fused(frames[k], ws, tz)
        if len(Bk) == N:
            start, C = k, np.array([b.mean(0) for b in Bk])
            break
    log(f"video: N={N} (frame0 fused) unit={unit:.0f} track start={start} blobs_ok={C is not None}")
    if C is None:
        Bk = frame_blobs_fused(frames[start], ws, tz)
        P = np.concatenate(Bk) if Bk else np.concatenate(B0)
        C = kmeans_init(P, np.array([b.mean(0) for b in B0]))
    ev["start"] = start
    d = np.linalg.norm(C - ev["release_xy"], axis=1)
    tgt = int(np.argmin(d))
    log(f"video: target idx {tgt} at {np.round(C[tgt],3)} dist_to_release={d[tgt]:.3f} second={np.sort(d)[1] if len(d)>1 else -1:.3f}")
    hist = track3(frames, ev["start"], C, unit, ws, tz)
    Cs = np.array([c for _, c in hist])
    N = Cs.shape[1]
    md = [min([np.linalg.norm(c[i] - c[j]) for i in range(N) for j in range(i + 1, N)] or [1]) for c in Cs]
    final = Cs[-1][tgt]
    others = [Cs[-1][j] for j in range(N) if j != tgt]
    log(f"video: final target {np.round(final,3)} min pair dist over track {min(md):.3f}")
    return {"target_xy": final, "others": others, "N": N, "hist": hist, "tgt": tgt, "event": ev,
            "min_pair": float(min(md)), "button": btn, "final_all": Cs[-1], "unit": unit}
