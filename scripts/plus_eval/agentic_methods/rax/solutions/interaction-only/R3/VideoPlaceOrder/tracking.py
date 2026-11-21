"""Track the identities of the target rings through the conditioning video.

Targets are visually identical; during the video pairs of them swap places (they rotate
about their midpoint). We track every ring frame to frame with constant-velocity prediction
and greedy assignment, so the identity of the ring the cube was released on can be followed
to its final position at the end of the video.
"""
import numpy as np
import perception as P

GATE_M = 0.07
MIN_PX = 25
FRAG_PX = 10


def frame_detections(f):
    files = {(x["camera"], x["kind"]): x["path"] for x in f.get("files", [])}
    if ("front", "rgb") not in files or ("front", "depth") not in files:
        return None
    try:
        rgb = P.load_rgb(files[("front", "rgb")]); dep = P.load_depth(files[("front", "depth")])
    except Exception:
        return None
    pts = P.world_points(P.target_mask(rgb), dep, f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    pts = pts[pts[:, 2] < 0.03]
    return [(c[:2], n) for c, n in P.cluster_xy(pts, 0.05) if n >= FRAG_PX]


def _assign(pred, dets, gate):
    pairs = []
    for i, q in enumerate(pred):
        for j, (d, n) in enumerate(dets):
            pairs.append((float(np.linalg.norm(q - d)), i, j))
    pairs.sort()
    used_i, used_j, out = set(), set(), {}
    for dist, i, j in pairs:
        if dist > gate:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); out[i] = j
    return out


def track(frames, log=None):
    """frames: ordered list (video_demo then execution). Returns dict with
    'pos' [T x K x 2] tracked positions, 'home' (first-frame positions), 'dets' list."""
    dets_all = [frame_detections(f) for f in frames]
    # initialise from the first frame with the most big detections
    k0 = 0
    best = -1
    for k, d in enumerate(dets_all[:20]):
        if d is None:
            continue
        nb = sum(1 for _, n in d if n >= MIN_PX)
        if nb > best:
            best, k0 = nb, k
        if nb >= 4:
            k0 = k; break
    init = [c for c, n in dets_all[k0] if n >= MIN_PX]
    K = len(init)
    pos = np.array(init, float); vel = np.zeros_like(pos); miss = np.zeros(K, int)
    hist = np.zeros((len(frames), K, 2))
    hist[:k0 + 1] = pos
    for k in range(k0 + 1, len(frames)):
        d = dets_all[k]
        if not d:
            pos = pos + vel * 0.5; vel *= 0.5; hist[k] = pos; continue
        pred = pos + vel
        big = [x for x in d if x[1] >= MIN_PX]
        a = _assign(pred, big, GATE_M)
        # fragments for the unmatched
        un = [i for i in range(K) if i not in a]
        if un:
            usedc = [big[j][0] for j in a.values()]
            frags = [x for x in d if x[1] < MIN_PX and all(np.linalg.norm(x[0] - u) > 0.03 for u in usedc)]
            a2 = _assign(pred[un], frags, GATE_M * 0.7)
        else:
            a2, frags = {}, []
        newpos = pred.copy()
        for i, j in a.items():
            newpos[i] = big[j][0]
        for ii, j in a2.items():
            newpos[un[ii]] = frags[j][0]
        matched = set(a) | {un[ii] for ii in a2}
        for i in range(K):
            if i in matched:
                vel[i] = 0.6 * (newpos[i] - pos[i]) + 0.4 * vel[i]; miss[i] = 0
            else:
                vel[i] *= 0.8; miss[i] += 1
        pos = newpos
        hist[k] = pos
    return {"pos": hist, "home": np.array(init), "k0": k0, "dets": dets_all}


def identity_at(hist_pos, k, xy, window=6, max_d=0.06):
    """Identity nearest to xy around frame k (median over a short window after k)."""
    T = hist_pos.shape[0]
    ks = range(max(0, k - 1), min(T, k + window))
    votes = {}
    for kk in ks:
        d = np.linalg.norm(hist_pos[kk] - np.asarray(xy[:2]), axis=1)
        i = int(np.argmin(d))
        if d[i] < max_d:
            votes[i] = votes.get(i, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)
