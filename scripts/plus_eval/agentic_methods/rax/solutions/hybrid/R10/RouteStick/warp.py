import numpy as np

def _perp(d):
    return np.array([-d[1], d[0]])

def warp_xy(p, A, B):
    """Map 2-D point p from chain A (n,2) to chain B (n,2): per-segment similarity,
    segment chosen by the projection parameter (clamped at the chain ends)."""
    n = len(A)
    best = None
    for i in range(n - 1):
        d = A[i+1] - A[i]; L2 = d @ d
        s = (p - A[i]) @ d / L2
        t = (p - A[i]) @ _perp(d) / L2
        inside = (0 <= s <= 1) or (i == 0 and s < 0) or (i == n-2 and s > 1)
        dist = abs(t) + (0 if 0 <= s <= 1 else min(abs(s), abs(s-1)))
        if inside and (best is None or dist < best[0]):
            best = (dist, i, s, t)
    if best is None:  # between segments due to a bend: nearest segment
        for i in range(n - 1):
            d = A[i+1] - A[i]; L2 = d @ d
            s = np.clip((p - A[i]) @ d / L2, 0, 1); t = (p - A[i]) @ _perp(d) / L2
            q = A[i] + s*d; dist = np.linalg.norm(p - q)
            if best is None or dist < best[0]:
                best = (dist, i, (p - A[i]) @ d / L2, t)
    _, i, s, t = best
    e = B[i+1] - B[i]
    return B[i] + s*e + t*_perp(e)

def warp_path(P, A, B):
    """P (N,3) world path; A, B (n,3) corresponding targets. z is kept."""
    out = P.copy()
    for k in range(len(P)):
        out[k, :2] = warp_xy(P[k, :2], A[:, :2], B[:, :2])
    return out
