"""Track identical objects through the conditioning video in the front camera using depth+saturation."""
import json, os
import numpy as np
from PIL import Image
from geom import depth_to_world

def load_frame(root, frame, cam="front"):
    files = {(f["camera"], f["kind"]): f["path"] for f in frame["files"]}
    rgb = np.array(Image.open(os.path.join(root, files[(cam, "rgb")])).convert("RGB"))
    dp = files.get((cam, "depth"))
    depth = None
    if dp:
        depth = np.array(Image.open(os.path.join(root, dp))).astype(np.float64) * 0.001
    return rgb, depth

def sat_of(rgb):
    rgbi = rgb.astype(np.int32)
    return rgbi.max(-1) - rgbi.min(-1)

def object_mask(rgb, z, zmin=0.006, zmax=0.08, sat_min=35):
    return np.isfinite(z) & (z > zmin) & (z < zmax) & (sat_of(rgb) > sat_min)

def components(mask, min_px=12):
    from geom import label
    lab, n = label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append(dict(u=xs.mean(), v=ys.mean(), n=len(xs), xs=xs, ys=ys))
    return out

def split_components(comps, z=None, log=print):
    """Split merged components using the median component size as the unit; drop huge structures."""
    if not comps:
        return np.zeros((0, 2)), []
    sizes = np.array([c["n"] for c in comps], float)
    unit = float(np.median(sizes))
    pts = []; foot = []
    for c in comps:
        k = int(round(c["n"] / unit))
        if k > 3:
            log(f"drop comp size {c['n']} unit {unit}"); continue
        k = max(1, k)
        P = np.stack([c["xs"], c["ys"]], 1).astype(float)
        if k == 1:
            pts.append(P.mean(0)); foot.append((c["xs"], c["ys"])); continue
        # k-means split
        rng = np.random.RandomState(0)
        C = P[rng.choice(len(P), k, replace=False)]
        # farthest point init
        C = [P[np.argmax(((P - P.mean(0)) ** 2).sum(1))]]
        while len(C) < k:
            d = np.min([((P - cc) ** 2).sum(1) for cc in C], axis=0)
            C.append(P[np.argmax(d)])
        C = np.array(C)
        for _ in range(15):
            a = ((P[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.array([P[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        for j in range(k):
            pts.append(C[j]); foot.append((c["xs"][a == j], c["ys"][a == j]))
    return np.array(pts), foot

def kmeans_step(prev, vel, mask, radius=12.0, iters=6, min_px=4, frozen=None):
    ys, xs = np.nonzero(mask)
    C = prev + vel
    if frozen is not None:
        C[frozen] = prev[frozen]
    counts = np.zeros(len(C), int)
    if len(xs) == 0:
        return C, counts
    Pp = np.stack([xs, ys], 1).astype(float)
    for _ in range(iters):
        d = ((Pp[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        if frozen is not None:
            d[:, frozen] = 1e9
        a = d.argmin(1)
        ok = d[np.arange(len(Pp)), a] < radius ** 2
        newC = C.copy()
        for k in range(len(C)):
            sel = ok & (a == k)
            counts[k] = sel.sum()
            if counts[k] >= min_px:
                newC[k] = Pp[sel].mean(0)
        C = newC
    return C, counts

def run(manifest_path, workspace="/workspace", log=print, cam="front"):
    m = json.load(open(manifest_path))
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    ex = [f for f in frames if f["phase"] == "execution"]
    vid = [f for f in frames if f["phase"] == "video_demo"]
    last = ex[-1] if ex else frames[-1]
    K = last["camera_intrinsics"][cam]["matrix"]; pose = last["camera_poses"][cam]
    seq = vid + ([ex[-1]] if ex else [])
    log(f"video frames {len(vid)} exec {len(ex)}")
    masks = []; zs = []; lifted = []
    chrom_ref = [None]
    for i, f in enumerate(seq):
        rgb, depth = load_frame(workspace, f, cam)
        P = depth_to_world(depth, K, pose)
        z = P[..., 2]
        masks.append(object_mask(rgb, z)); zs.append(z.astype(np.float32))
        if i < len(seq) * 0.4:
            rg = rgb.astype(np.float32); ch = rg / (rg.sum(-1, keepdims=True) + 1e-6)
            L = (sat_of(rgb) > 35) & (z > 0.05) & (z < 0.4) & (P[..., 0] > -0.45)
            ys_, xs_ = np.nonzero(L)
            lifted.append((ch[ys_, xs_], P[ys_, xs_, :2]))
        else:
            lifted.append(None)
    comps = components(masks[0])
    P0 = depth_to_world(load_frame(workspace, seq[0], cam)[1], K, pose)
    keep = []
    for c in comps:
        W = P0[c["ys"], c["xs"]]
        if W[:, 0].mean() < -0.45 or np.percentile(W[:, 2], 95) > 0.07:
            log(f"drop init comp n={c['n']} xy={W[:, :2].mean(0).round(3)} ztop={np.percentile(W[:, 2], 95):.3f}")
            continue
        keep.append(c)
    comps = keep
    pts, foot = split_components(comps, log=log)
    K_ = len(pts)
    init_xy = np.array([P0[ys, xs, :2].mean(0) for (xs, ys) in foot]) if K_ else np.zeros((0, 2))
    rgb0 = load_frame(workspace, seq[0], cam)[0]
    allm = np.zeros(masks[0].shape, bool)
    for (xs, ys) in foot:
        allm[ys, xs] = True
    c0 = rgb0[allm].astype(float)
    chrom = np.median(c0 / (c0.sum(1, keepdims=True) + 1e-6), 0) if len(c0) else np.array([1/3] * 3)
    log(f"init comps {len(comps)} sizes {[c['n'] for c in comps]} -> {K_} objects")
    # --- pick detection: footprint pixels become table-level (empty) ---
    T = len(seq)
    empty = np.zeros((T, K_))
    for t in range(T):
        z = zs[t]
        for k, (xs, ys) in enumerate(foot):
            empty[t, k] = np.mean(z[ys, xs] < 0.003)
    # smooth over 3 frames
    ker = np.ones(3) / 3
    sm = np.stack([np.convolve(empty[:, k], ker, mode="same") for k in range(K_)], 1)
    # target = first object whose footprint becomes empty (the pick precedes the shuffle)
    hit = sm > 0.5
    first_t = np.array([np.argmax(hit[:, k]) if hit[:, k].any() else T + 1 for k in range(K_)])
    target_empty = int(first_t.argmin())
    t_pick = int(first_t[target_empty]) if first_t[target_empty] <= T else None
    # lifted interval: from t_pick until footprint is occupied again
    t_back = t_pick
    if t_pick is not None:
        t_back = t_pick
        while t_back < T and sm[t_back, target_empty] > 0.5:
            t_back += 1
    order = np.argsort(first_t)
    gap_e = int(first_t[order[1]] - first_t[order[0]]) if K_ > 1 else 99
    raw_empty = (init_xy[target_empty].tolist(), int(t_pick), int(t_back)) if (K_ and t_pick is not None and gap_e >= 6) else None
    # --- lifted-object cue: colour-matched pixels well above the table ---
    lift_cnt = np.zeros(T, int); lift_xy = np.zeros((T, 2))
    for t in range(T):
        if lifted[t] is None:
            continue
        ch, xy = lifted[t]
        if len(ch) == 0:
            continue
        sel = np.abs(ch - chrom).sum(-1) < 0.06
        lift_cnt[t] = sel.sum()
        if sel.sum():
            lift_xy[t] = xy[sel].mean(0)
    lt = None; lt_end = None
    for t in range(T - 1):
        if 30 <= lift_cnt[t] <= 600 and 30 <= lift_cnt[t + 1] <= 600 and np.linalg.norm(lift_xy[t] - lift_xy[t + 1]) < 0.03:
            lt = t; break
    lift_target = None
    if lt is not None and K_:
        dd = np.linalg.norm(init_xy - lift_xy[lt], axis=1)
        lift_target = int(dd.argmin())
        lt_end = lt
        while lt_end < T and lift_cnt[lt_end] >= 10:
            lt_end += 1
        log(f"lift cue: t {lt}-{lt_end} xy {lift_xy[lt].round(3)} nearest obj {lift_target} dist {dd.min():.3f}")
        if dd.min() > 0.06:
            lift_target = None
    if gap_e < 6 or t_pick is None:
        if lift_target is not None:
            log(f"empty cue ambiguous (gap {gap_e}); using lift cue target {lift_target}")
            target_empty = lift_target; t_pick = lt; t_back = lt_end + 3
    elif lift_target is not None and lift_target != target_empty:
        log(f"WARNING cues disagree: empty {target_empty} vs lift {lift_target}; keeping empty cue")
    score = first_t
    log(f"first-empty times {first_t.tolist()} -> target {target_empty} lifted [{t_pick},{t_back}) runner-up gap {int(first_t[order[1]] - first_t[order[0]]) if K_ > 1 else -1}")
    # --- tracking ---
    pos = [pts.copy()]; cnts = [np.array([len(f[0]) for f in foot])]
    vel = np.zeros_like(pts)
    for t in range(1, T):
        frozen = np.zeros(K_, bool)
        # while target footprint is empty (lifted), freeze its track
        if t_pick is not None and t_pick <= t < t_back:
            frozen[target_empty] = True
        C, cnt = kmeans_step(pos[-1], vel * 0.6, masks[t], frozen=frozen)
        vel = np.where(cnt[:, None] >= 4, C - pos[-1], vel * 0.8)
        vel[frozen] = 0
        pos.append(C); cnts.append(cnt)
    pos = np.array(pos); cnts = np.array(cnts)
    # last-seen world xy of every track (robust to final-frame occlusion)
    tk = target_empty if t_pick is not None else None
    last_seen = {}
    for k in range(K_):
        for t in range(T - 1, max(-1, T - 200), -1):
            if cnts[t, k] >= 8:
                last_seen[k] = t; break
    info_ls = {}
    for k, t in last_seen.items():
        try:
            rgb, depth = load_frame(workspace, seq[t], cam)
            P = depth_to_world(depth, K, pose)
            mk = masks[t]; ys, xs = np.nonzero(mk)
            d = np.hypot(xs - pos[t, k, 0], ys - pos[t, k, 1])
            sel = d < 7
            if sel.sum() >= 4:
                info_ls[k] = (t, P[ys[sel], xs[sel], :2].mean(0).tolist())
        except Exception as e:
            log("last-seen failed", repr(e))
        if k != tk and len(info_ls) > 0 and tk is not None and tk in info_ls:
            pass
    fm, tfm = first_mover(pos)
    log(f"first mover {fm} at t {tfm}; empty-target {target_empty}")
    if K_ > 1:
        gap = int(np.sort(first_t)[1] - np.sort(first_t)[0])
        if t_pick is not None and gap < 6 and lift_target is None and fm != target_empty:
            log(f"ambiguous empty detection (gap {gap}); falling back to first mover {fm}")
            t_pick = None
    info = dict(target_empty=target_empty, empty_first=first_t.tolist(), t_back=t_back, t_pick=t_pick, first_mover=fm, t_first=tfm,
                final=pos[-1].tolist(), min_counts=cnts.min(0).tolist(),
                last_seen={int(k): v for k, v in info_ls.items()}, final_counts=cnts[-1].tolist(),
                lift=(dict(t=int(lt), t_end=int(lt_end), xy=lift_xy[lt].tolist()) if lt is not None else None),
                raw_empty=raw_empty,
                empty_xy=(init_xy[target_empty].tolist() if (K_ and t_pick is not None) else None))
    return pos, cnts, info, (K, pose), seq

def first_mover(pos, thresh=5.0):
    disp = np.linalg.norm(pos - pos[0][None], axis=2)
    for t in range(len(pos)):
        k = int(disp[t].argmax())
        if disp[t, k] > thresh:
            return k, t
    return int(disp[-1].argmax()), len(pos) - 1
