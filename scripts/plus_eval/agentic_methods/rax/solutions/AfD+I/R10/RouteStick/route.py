"""Parse the conditioning video's robot-state trajectory into a route."""
import numpy as np

def dwell_points(S, z_draw_tol=0.02, still=0.004, min_len=3):
    """S: (N,>=3) TCP positions in video order. Returns list of (start,end,xy) dwell clusters at drawing height."""
    P = np.asarray(S)[:, :3]
    zmin = np.min(P[:, 2])
    low = P[:, 2] < zmin + z_draw_tol
    sp = np.r_[np.inf, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)]
    st = low & (sp < still)
    dw = []
    i = 0; N = len(P)
    while i < N:
        if st[i]:
            j = i
            while j + 1 < N and st[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                xy = P[i:j + 1, :2].mean(0)
                if dw and np.linalg.norm(dw[-1][2] - xy) < 0.03:
                    a, _, _ = dw[-1]
                    dw[-1] = (a, j, P[a:j + 1, :2].mean(0))
                else:
                    dw.append((i, j, xy))
            i = j + 1
        else:
            i += 1
    return dw, zmin

def dwell_points_t(P, T, discs=None, z_tol=0.02, vthr=0.06):
    """Time-based dwell detection: low TCP speed at drawing height, optionally snapped to discs."""
    P = np.asarray(P)[:, :3]; T = np.asarray(T, float)
    zmin = float(np.min(P[:, 2]))
    dt = np.maximum(np.diff(T), 1e-3)
    v = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) / dt
    sp = np.r_[v[:1], np.minimum(v[:-1], v[1:]) if len(v) > 1 else v[:0], v[-1:]] if len(v) else np.zeros(len(P))
    sp = sp[:len(P)]
    low = P[:, 2] < zmin + z_tol
    cand = low & (sp < vthr)
    groups = []
    i = 0; N = len(P)
    while i < N:
        if cand[i]:
            j = i
            while j + 1 < N and cand[j + 1]:
                j += 1
            k = i + int(np.argmin(sp[i:j + 1]))
            groups.append([i, j, P[k, :2].copy()])
            i = j + 1
        else:
            i += 1
    if discs is not None and len(discs) >= 3:
        D = np.asarray(discs)
        keep = []
        for g in groups:
            d = np.linalg.norm(D - g[2], axis=1)
            if d.min() < 0.035:
                g.append(int(np.argmin(d)))
                keep.append(g)
        groups = keep
    # merge consecutive groups at the same place
    out = []
    for g in groups:
        if out and np.linalg.norm(out[-1][2] - g[2]) < 0.035:
            out[-1][1] = g[1]
        else:
            out.append(g)
    return [(g[0], g[1], g[2]) for g in out], zmin

def parse_route(S, T=None, discs=None):
    """Return dict with targets (video xy clustered), visit sequence indices, and side per move.
    side +1 means arc toward +x (camera side), -1 toward -x (robot side)."""
    P = np.asarray(S)[:, :3]
    if T is not None:
        dw, zmin = dwell_points_t(P, T, discs)
    else:
        dw, zmin = dwell_points(P)
    # cluster dwell xy into distinct targets
    targets = []
    seq = []
    for (_, _, xy) in dw:
        k = None
        for t, c in enumerate(targets):
            if np.linalg.norm(c - xy) < 0.04:
                k = t; break
        if k is None:
            targets.append(xy.copy()); k = len(targets) - 1
        seq.append(k)
    # refine each target as the median of its dwell points, ignoring the initial approach dwell
    for t in range(len(targets)):
        pts = [dw[i][2] for i in range(len(dw)) if seq[i] == t and (i > 0 or seq.count(t) == 1)]
        targets[t] = np.median(np.array(pts), 0)
    moves = []
    for m in range(1, len(dw)):
        a = dw[m - 1][1]; b = dw[m][0]
        A = dw[m - 1][2]; B = dw[m][2]
        seg = P[a:b + 1, :2]
        ch = B - A; L = np.linalg.norm(ch) + 1e-9
        n = np.array([-ch[1], ch[0]]) / L
        dev = (seg - A) @ n
        k = int(np.argmax(np.abs(dev)))
        h = dev[k]
        # express side as world x direction of the deviation
        dx = (seg[k] - (A + ((seg[k] - A) @ ch / L) * ch / L))[0]
        moves.append(dict(frm=seq[m - 1], to=seq[m], h=float(abs(h)), side=int(np.sign(dx)) or 1,
                          steps=int(b - a), peak=seg[k].tolist()))
    # also record the trailing motion after last dwell (incomplete move?)
    tail = None
    if dw and dw[-1][1] < len(P) - 1:
        seg = P[dw[-1][1]:, :2]
        tail = float(np.max(np.linalg.norm(seg - dw[-1][2], axis=1)))
    return dict(targets=[t.tolist() for t in targets], seq=seq, moves=moves, z_draw=float(zmin),
                dwells=[(int(a), int(b), xy.tolist()) for a, b, xy in dw], tail=tail)
