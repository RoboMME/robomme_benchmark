"""Parse the conditioning video (front camera frames) into the demonstrated route:
an ordered list of visited target indices and, for every move between two
consecutive targets, which side of the line between them the drawn trail took
in the front image ('near' = below the line in the image = towards the camera,
'far' = above it)."""
import numpy as np
from vision import detect_targets, red_counts, label

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def wood_mask(rgb):
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1)
    mn = a.min(-1)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    # hue in degrees (only need the orange band)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return (h >= 6) & (h <= 34) & (s >= 0.43)


def white_mask(rgb):
    a = rgb.astype(np.int32)
    mn = a.min(-1)
    mx = a.max(-1)
    return (mn > 165) & (mx - mn < 40)


def dilate(mask, it=1):
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=it)
    m = mask.copy()
    for _ in range(it):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m


def visit_sequence(fronts, targets, radius=None, min_px=6):
    """fronts: list of HxWx3 arrays in time order. Returns list of (target_idx, frame_pos)."""
    T = targets
    if radius is None:
        sp = min(np.hypot(T[i + 1][0] - T[i][0], T[i + 1][1] - T[i][1]) for i in range(len(T) - 1)) if len(T) > 1 else 20
        radius = max(3.0, min(8.0, 0.3 * sp))
    base = np.array(red_counts(fronts[0], T, radius))
    seq = []
    counts = []
    for k, fr in enumerate(fronts):
        c = np.array(red_counts(fr, T, radius)) - base
        counts.append(c.tolist())
        kk = int(np.argmax(c))
        if c[kk] >= min_px and (not seq or seq[-1][0] != kk):
            seq.append((kk, k))
    return seq, counts


def classify_move(fronts, bg, targets, i, j, ka, kb, debug=None):
    """Side of the trail drawn while moving from target i (red onset at frame ka)
    to target j (onset kb)."""
    Ws = np.array([white_mask(fronts[k]) & ~bg for k in range(ka, max(kb, ka + 1))])
    n = len(Ws)
    anyw = Ws.any(0)
    first = np.where(anyw, Ws.argmax(0), n)
    cnt = Ws.sum(0)
    span = n - first
    pers = np.where(span > 0, cnt / np.maximum(span, 1), 0)
    m = Ws[-1] & (pers >= 0.75) & (span >= max(3, 0.12 * n))
    P = np.array(targets[i]); Q = np.array(targets[j]); d = Q - P; L = float(np.linalg.norm(d))
    u = d / L
    nrm = np.array([-u[1], u[0]])
    if nrm[1] < 0:
        nrm = -nrm  # points down in the image (near side)
    ys, xs = np.nonzero(m)
    v = np.stack([xs, ys], 1) - P
    t = v @ u / L
    s = v @ nrm
    sel = (t > 0.08) & (t < 0.92)
    for (tx, ty) in targets:
        sel &= ((xs - tx) ** 2 + (ys - ty) ** 2 > 36)
    # The hand is drawn above the stick tip in the image, so only a short band
    # above the line is trusted for far-side trail; near-side trail is unoccluded.
    near = int(((s > 1.5) & (s < 1.0 * L) & sel).sum())
    far = int(((s < -1.5) & (s > -0.5 * L) & sel).sum())
    far_loose = int(((s < -1.5) & (s > -0.8 * L) & sel).sum())
    res = {"near": near, "far": far, "far_loose": far_loose}
    if near >= 6 and near >= 0.6 * far:
        side, conf = "near", min(1.0, near / 20.0)
    elif far >= 4 and far > 1.5 * near:
        side, conf = "far", min(1.0, far / 15.0)
    elif near < 3:
        side, conf = "far", 0.3  # far-side trails are often hidden behind the obstacle
    else:
        side, conf = ("near" if near >= far else "far"), 0.2
    if debug is not None:
        debug.append(dict(i=i, j=j, ka=ka, kb=kb, counts=res, side=side, conf=conf))
    return side, conf


def parse_video(fronts, top_exclude=20):
    """Return dict(targets, visits=[idx...], moves=[(i, j, side, conf)], debug)."""
    targets = detect_targets(fronts[0], top_exclude=top_exclude)
    seq, _ = visit_sequence(fronts, targets)
    bg = dilate(white_mask(fronts[0]), 1)
    dbg = []
    moves = []
    visits = [seq[0][0]] if seq else []
    for (i, ka), (j, kb) in zip(seq, seq[1:]):
        if abs(i - j) == 1:
            side, conf = classify_move(fronts, bg, targets, i, j, ka, kb, dbg)
            moves.append((i, j, side, conf))
            visits.append(j)
        else:
            # a visit was missed (occlusion?): walk through the intermediate targets,
            # classifying each sub-move on the same frames
            step = 1 if j > i else -1
            for k in range(i, j, step):
                side, conf = classify_move(fronts, bg, targets, k, k + step, ka, kb, dbg)
                moves.append((k, k + step, side, conf * 0.5))
                visits.append(k + step)
    return dict(targets=targets, visits=visits, onsets=[s[1] for s in seq], moves=moves, debug=dbg)
