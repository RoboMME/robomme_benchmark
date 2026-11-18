"""World-space tracker using per-frame RGB-D of both cameras in the conditioning video.
Picked object = where the robot released it (finger opening + EE position in recorded state)."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from robot import WS, world_map, log
from tracker import Seg

ZMIN, ZMAX = 0.004, 0.07


def _load(path):
    with Image.open(WS / path) as im:
        return np.array(im.convert("RGB")) if im.mode != "I;16" and im.mode != "I" else np.array(im)


def load_frames(man):
    frames = [f for f in sorted(man["frames"], key=lambda f: f["frame_index"]) if f["phase"] == "video_demo"]
    return frames


def frame_points(fr, seg_front, seg_wrist, cams=("front", "wrist"), exclude=None):
    files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
    pts = []
    for cam in cams:
        if (cam, "rgb") not in files or (cam, "depth") not in files:
            continue
        with Image.open(WS / files[(cam, "rgb")]) as im:
            rgb = np.array(im.convert("RGB"))
        with Image.open(WS / files[(cam, "depth")]) as im:
            dep = np.array(im)
        seg = seg_front if cam == "front" else seg_wrist
        m = seg.mask(rgb)
        if cam == "wrist":
            m[190:, :100] = False; m[190:, 200:] = False   # own fingers
        if not m.any():
            continue
        W = world_map(fr, dep, cam)
        P = W[m]
        P = P[~np.isnan(P).any(1)]
        P = P[(P[:, 2] > ZMIN) & (P[:, 2] < ZMAX)]
        if cam == "wrist" and len(P) > 1500:
            P = P[:: int(np.ceil(len(P) / 1500))]
        if exclude is not None and len(P):
            P = P[np.linalg.norm(P[:, :2] - np.asarray(exclude)[None, :2], axis=1) > 0.045]
        pts.append(P)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def grid_components(P, cell=0.004, min_pts=8):
    if len(P) == 0:
        return []
    lo = P[:, :2].min(0) - 0.01
    ij = np.floor((P[:, :2] - lo) / cell).astype(int)
    H, Wd = ij.max(0) + 2
    occ = np.zeros((H, Wd), bool); occ[ij[:, 0], ij[:, 1]] = True
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, n = ndimage.label(occ)
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        sel = pl == k
        if sel.sum() >= min_pts:
            out.append((np.median(P[sel, :2], 0), int(sel.sum())))
    return out


def kmeans_xy(P, C, iters=5, radius=0.03):
    C = C.copy(); cnt = np.zeros(len(C), int)
    if len(P) == 0 or len(C) == 0:
        return C, cnt
    for _ in range(iters):
        d = ((P[:, None, :2] - C[None]) ** 2).sum(-1)
        a = d.argmin(1); ok = d.min(1) < radius ** 2
        cnt = np.bincount(a[ok], minlength=len(C))
        for k in range(len(C)):
            sel = ok & (a == k)
            if sel.sum() >= 5:
                C[k] = P[sel, :2].mean(0)
    return C, cnt


def events(frames):
    S = np.array([f["state"] for f in frames], float)
    op = S[:, 6] + S[:, 7]
    full = np.max(op)
    tg = next((i for i in range(len(op)) if op[i] < full - 0.01), None)
    if tg is None:
        return None
    tr = next((i for i in range(tg, len(op)) if op[i] > full - 0.008), None)
    if tr is None:
        return None
    # release position: EE xy at release
    place = S[tr, :3]
    # arm parked: first frame after tr where EE stops moving for 4 frames
    post = None
    for i in range(tr + 1, len(S) - 4):
        if np.abs(S[i:i + 4, :3] - S[i, :3]).max() < 5e-4 and S[i, 2] > 0.12:
            post = i; break
    return {"tg": tg, "tr": tr, "place": place, "grasp": S[tg, :3], "post": post}


def identify_world(meta, seg_front=None, log=log, exclude=None):
    seq = meta.get("frame_sequence") or {}
    man = json.loads((WS / seq["manifest_path"]).read_text())
    frames = load_frames(man)
    ev = events(frames)
    if ev is None or ev["post"] is None:
        log("[wtrack] no grasp/release events found")
        return None
    log(f"[wtrack] grasp f{ev['tg']} at {ev['grasp'].round(3).tolist()} release f{ev['tr']} at {ev['place'].round(3).tolist()} parked f{ev['post']}")
    if seg_front is None:
        seg_front = Seg()
        pre = []
        for fr in frames[:3]:
            files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
            with Image.open(WS / files[("front", "rgb")]) as im:
                pre.append(np.array(im.convert("RGB")))
        seg_front.calibrate(pre)
    seg_wrist = Seg(); seg_wrist.obj_hue = seg_front.obj_hue
    post = ev["post"]
    P = frame_points(frames[post], seg_front, seg_wrist, exclude=exclude)
    comps = grid_components(P)
    if not comps:
        log("[wtrack] no objects at parked frame"); return None
    C = np.array([c for c, n in comps], float)
    sizes = np.array([n for c, n in comps])
    d = np.linalg.norm(C - ev["place"][:2][None], axis=1)
    k = int(d.argmin())
    # tie-breaker: only the picked object moved between t=0 (arm high) and the parked frame
    try:
        P0 = frame_points(frames[0], seg_front, seg_wrist, exclude=exclude)
        C0 = np.array([c for c, n in grid_components(P0)], float).reshape(-1, 2)
        if len(C0) and len(C) > 1:
            moved = np.array([np.linalg.norm(C0 - c[None], axis=1).min() for c in C])
            order = np.argsort(d)
            a, b = order[0], order[1]
            log(f"[wtrack] release dists {d[a]:.3f}/{d[b]:.3f} moved {moved[a]:.3f}/{moved[b]:.3f}")
            if d[b] < 1.5 * d[a] and moved[b] > moved[a] + 0.01:
                k = int(b); log("[wtrack] tie-break: choosing the moved candidate")
    except Exception as e:
        log("[wtrack] tie-break skipped", e)
    log(f"[wtrack] {len(C)} objects at parked frame; target #{k} at {C[k].round(3).tolist()} (d to release {d[k]:.3f})")
    if d[k] > 0.05:
        # target hidden at parked frame: create a track at the release point
        C = np.vstack([C, ev["place"][:2]]); k = len(C) - 1
        log("[wtrack] target not visible at parked frame; seeded at release point")
    lost = np.zeros(len(C), int)
    tgt_hist = [C[k].copy()]
    for i in range(post + 1, len(frames)):
        P = frame_points(frames[i], seg_front, seg_wrist, exclude=exclude)
        Cn, cnt = kmeans_xy(P, C)
        sup = cnt >= 5
        Cn[~sup] = C[~sup]
        lost[~sup] += 1; lost[sup] = 0
        # births: components far from all supported tracks
        if len(P):
            dmin = np.sqrt(((P[:, None, :2] - Cn[None]) ** 2).sum(-1)).min(1)
            far = P[dmin > 0.035]
            for c, n in grid_components(far, min_pts=15):
                # re-associate to a lost track if one is near, else new track
                lost_idx = np.where(lost > 0)[0]
                if len(lost_idx):
                    dl = np.linalg.norm(Cn[lost_idx] - c[None], axis=1)
                    j = lost_idx[dl.argmin()]
                    if dl.min() < 0.08:
                        Cn[j] = c; lost[j] = 0; continue
                Cn = np.vstack([Cn, c]); lost = np.append(lost, 0)
        C = Cn
        tgt_hist.append(C[k].copy())
    log(f"[wtrack] final target {C[k].round(3).tolist()} lost={lost[k]} tracks={len(C)} all={np.round(C, 3).tolist()}")
    return {"target_xy": C[k].tolist(), "lost": int(lost[k]), "all_xy": C.tolist(), "events":
            {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in ev.items()},
            "hue": seg_front.obj_hue, "hist": np.array(tgt_hist).tolist()}
