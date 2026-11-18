"""Parse the conditioning video (front RGB frames) into a route.

Route = ordered list of disk indices visited (indices 0..4 ordered left->right in the
front image) and, for each consecutive move, the side of the pillar passed:
'front' (toward the front camera = below the disk line in the image) or 'back'.

Pure functions over numpy arrays; no I/O except an optional loader passed in.
Validated offline on the 3 released demos (22 moves, all correct).
"""
import numpy as np

from vision import hsv, red_mask, white_mask, disk_candidates, choose_row

try:
    from scipy.ndimage import binary_dilation as _bd

    def dilate(m, it=1):
        return _bd(m, iterations=it)
except Exception:  # pragma: no cover
    def dilate(m, it=1):
        m = m.copy()
        for _ in range(it):
            p = np.pad(m, 1)
            acc = np.zeros_like(m)
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    acc |= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
            m = acc
        return m

RED_WIN_PX = 11        # radius around disk centroid for red counting (demo red blob ~70-85 px)
RED_MIN_PX = 12        # red pixels needed to call a disk red (tested: 0 when not red, >=19 when red)
TRAIL_VMIN = 165       # trail-like pixel brightness
TRAIL_SMAX = 0.22
FRONT_MIN_PX = 6       # appearance px below line for FRONT (demos: back 0, front 13..83)
LAT_MIN_PX, LAT_MAX_PX = 4, 40
ALONG_LO, ALONG_HI = 0.2, 0.8


def detect_disks(front0):
    cands = disk_candidates(front0)
    r = choose_row(cands)
    if r is None:
        return None, cands
    D = np.array([[cands[i]['u'], cands[i]['v']] for i in r[0]], float)
    return D, cands


def red_counts(img, red0, D):
    rm = red_mask(img) & ~red0
    H, W = rm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for u, v in D:
        out.append(int((rm & ((xx - u) ** 2 + (yy - v) ** 2 < RED_WIN_PX ** 2)).sum()))
    return out


def trail_like(a):
    h, s, v = hsv(a)
    return (s < TRAIL_SMAX) & (v > TRAIL_VMIN)


def parse_route(frames, D, front0=None, log=print):
    """frames: list of HxWx3 uint8 arrays (video order). D: (n,2) disk pixel centroids.

    Returns dict(sequence=[...], moves=[(a,b,side,evidence)...], arrivals=[(t,k)]).
    """
    if front0 is None:
        front0 = frames[0]
    red0 = red_mask(front0)
    n = len(D)
    isred = []
    counts_all = []
    for a in frames:
        c = red_counts(a, red0, D)
        counts_all.append(c)
        isred.append([x >= RED_MIN_PX for x in c])
    arrivals = []
    for t in range(len(frames)):
        for k in range(n):
            if isred[t][k] and (t == 0 or not isred[t - 1][k]):
                arrivals.append((t, k))
    # dedupe consecutive same-disk arrivals (flicker / occlusion)
    ded = []
    for t, k in arrivals:
        if ded and ded[-1][1] == k:
            continue
        ded.append((t, k))
    # expand non-adjacent jumps into adjacent steps (missed arrival by occlusion)
    exp = []
    for i, (t, k) in enumerate(ded):
        if exp and abs(k - exp[-1][1]) > 1:
            k0 = exp[-1][1]
            step = 1 if k > k0 else -1
            log(f"[video] non-adjacent jump {k0}->{k} at t={t}; inserting intermediates")
            for kk in range(k0 + step, k, step):
                exp.append((t, kk))
        exp.append((t, k))
    arrivals = exp
    TW = [trail_like(a) for a in frames]
    H, W = frames[0].shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    P = np.stack([xx, yy], -1).astype(float)
    moves = []
    for i in range(len(arrivals) - 1):
        (t0, a), (t1, b) = arrivals[i], arrivals[i + 1]
        A, B = D[a], D[b]
        L = np.linalg.norm(B - A)
        if L < 1e-6:
            continue
        d = (B - A) / L
        nrm = np.array([-d[1], d[0]])
        if nrm[1] < 0:
            nrm = -nrm  # points down in the image = toward the front camera
        along = (P - A) @ d
        lat = (P - (A + B) / 2) @ nrm
        corr = (along > ALONG_LO * L) & (along < ALONG_HI * L)
        below_region = corr & (lat > LAT_MIN_PX) & (lat < LAT_MAX_PX)
        acc = np.zeros((H, W), bool)
        tend = t1 if t1 > t0 else min(t0 + 1, len(frames) - 1)
        for t in range(t0 + 1, min(tend + 1, len(frames))):
            acc |= TW[t] & ~dilate(TW[t - 1], 1)
        ev = int((acc & below_region).sum())
        side = 'front' if ev >= FRONT_MIN_PX else 'back'
        moves.append((a, b, side, ev))
    seq = [k for _, k in arrivals]
    return dict(sequence=seq, moves=moves, arrivals=arrivals,
                red_counts_last=counts_all[-1] if counts_all else None)
