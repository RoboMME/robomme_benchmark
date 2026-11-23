import json, numpy as np
from pathlib import Path
from vision import load_rgb, load_depth
from seg import find_button, object_points, object_blobs, merge_buttons

ROOT = Path("/workspace")

def frame_files(f):
    return {(x["camera"], x["kind"]): x["path"] for x in f["files"]}

def load_cam(f, cam):
    files = frame_files(f)
    if (cam, "rgb") not in files or (cam, "depth") not in files:
        return None
    return (load_rgb(ROOT / files[(cam, "rgb")]), load_depth(ROOT / files[(cam, "depth")]),
            f["camera_intrinsics"][cam], f["camera_poses"][cam])

def kmeans_step(xy, centers, iters=5, gate=0.06, w=None):
    C = centers.copy()
    if w is None:
        w = np.ones(len(xy))
    if len(xy) == 0:
        return C
    for _ in range(iters):
        d = np.linalg.norm(xy[:, None, :] - C[None], axis=2)
        a = d.argmin(1); dm = d.min(1)
        newC = C.copy()
        for k in range(len(C)):
            sel = (a == k) & (dm < gate)
            if w[sel].sum() >= 8:
                newC[k] = (xy[sel] * w[sel, None]).sum(0) / w[sel].sum()
        C = newC
    return C

def wrist_ok(f):
    s = f["state"]
    return s[2] > 0.15 and (s[6] + s[7]) > 0.07

def frame_xy_w(f, button, use_wrist=True, ww=0.3):
    xs, ws = [], []
    c = load_cam(f, "front")
    if c is not None:
        p, _ = object_points(*c, button=button)
        xs.append(p[:, :2]); ws.append(np.ones(len(p)))
    if use_wrist and wrist_ok(f):
        c = load_cam(f, "wrist")
        if c is not None:
            p, _ = object_points(*c, button=button)
            xs.append(p[:, :2]); ws.append(ww * np.ones(len(p)))
    if not xs:
        return np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ws)

def count_objects(f, button, log):
    best = []
    for cam in ("front", "wrist"):
        if cam == "wrist" and not wrist_ok(f):
            continue
        c = load_cam(f, cam)
        if c is None:
            continue
        bl = object_blobs(*c, button=button)
        if not bl:
            continue
        sizes = np.array([b["n"] for b in bl]); med = np.median(sizes)
        k = int(sum(max(1, round(s / med)) for s in sizes))
        log(f"count {cam}: blobs {len(bl)} -> K {k} sizes {sorted(sizes.tolist())}")
        if len(bl) == k and len(bl) >= len(best):
            best = [b["xy"] for b in bl]
        elif not best:
            best = [b["xy"] for b in bl]
    return np.array(best)

def analyze_video(manifest_path, log=print, use_wrist=True):
    p = Path(manifest_path)
    m = json.loads((p if p.is_absolute() else ROOT / p).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f["phase"] == "video_demo"]
    ex = [f for f in frames if f["phase"] != "video_demo"]
    allf = vid + ex[:1]
    S = np.array([f["state"] for f in allf])
    grip = S[:, 6] + S[:, 7]
    closed = np.where(grip < 0.06)[0]
    if len(closed) == 0:
        raise RuntimeError("no grasp found in video")
    g0 = closed[0]
    rel = g0 + np.where(grip[g0:] > 0.07)[0]
    r0 = rel[0] if len(rel) else g0
    grasp_xy = S[g0, :2]; release_xy = S[r0, :2]
    up = r0 + np.where(S[r0:, 2] > 0.15)[0]
    s0 = up[0] if len(up) else r0
    c0 = load_cam(allf[0], "front")
    button = find_button(*c0) if c0 is not None else None
    if wrist_ok(allf[0]):
        cw = load_cam(allf[0], "wrist")
        if cw is not None:
            button = merge_buttons(button, find_button(*cw))
    log(f"video: grasp frame {g0} xyz {S[g0,:3].round(3)}, release frame {r0} xy {release_xy.round(3)} z {S[r0,2]:.3f}, start {s0}/{len(allf)}; button {None if button is None else button['xy'].round(3)}")
    C = count_objects(allf[0], button, log)
    K = len(C)
    if K == 0:
        raise RuntimeError("no objects in frame 0")
    xy, w = frame_xy_w(allf[s0], button, use_wrist)
    C = kmeans_step(xy, C)
    tgt = int(np.linalg.norm(C - release_xy, axis=1).argmin())
    log(f"target init {C[tgt].round(3)} dist to release {np.linalg.norm(C[tgt]-release_xy):.3f}")
    V = np.zeros_like(C)
    hist = [(s0, C.copy())]
    min_nn = 9.0
    for i in range(s0 + 1, len(allf)):
        xy, w = frame_xy_w(allf[i], button, use_wrist)
        if len(xy) < 20:
            hist.append((i, C.copy())); continue
        pred = C + V
        Cn = kmeans_step(xy, pred, w=w)
        V = 0.5 * V + 0.5 * (Cn - C)
        C = Cn
        hist.append((i, C.copy()))
        dd = np.linalg.norm(C - C[tgt], axis=1); dd[tgt] = 9
        min_nn = min(min_nn, float(dd.min()))
    return dict(K=K, target=tgt, final_xy=C[tgt], centers=C, hist=hist, grasp_xy=grasp_xy,
                release_xy=release_xy, grasp_z=float(S[g0, 2]), release_z=float(S[r0, 2]), start=s0, n=len(allf),
                min_nn=min_nn, button=button)
