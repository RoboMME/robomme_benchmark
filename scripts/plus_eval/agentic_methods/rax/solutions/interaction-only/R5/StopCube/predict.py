import numpy as np

def closest_on_seg(a, b, p):
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-12: return 0.0, float(np.linalg.norm(p - a))
    s = float(np.clip((p - a) @ ab / L2, 0, 1))
    return s, float(np.linalg.norm(a + s * ab - p))

def linear_pred(hist, T):
    """hist: list of (step, xy). Return (tau, dmin) using last two consecutive samples."""
    if len(hist) < 2: return None
    (s0, p0), (s1, p1) = hist[-2], hist[-1]
    dt = s1 - s0
    if dt <= 0 or dt > 2: return None
    v = (p1 - p0) / dt; vv = float(v @ v)
    if vv < 1e-8: return None
    tau = -float((p1 - T) @ v) / vv
    return tau, float(np.linalg.norm(p1 - T + tau * v))

def template_pred(hist, T, period, max_gap=2):
    """Match current position to the previous orbit; return (tau, dmin_template) or None."""
    if period is None or len(hist) < 10: return None
    s_now, p = hist[-1]
    if len(hist) >= 2 and hist[-2][0] >= s_now - 2:
        vcur = p - hist[-2][1]
    else:
        return None
    lo, hi = s_now - 1.5 * period, s_now - 0.5 * period
    best = None
    for k in range(len(hist) - 1):
        (sa, a), (sb, b) = hist[k], hist[k + 1]
        if sa < lo or sb > hi or sb - sa > max_gap: continue
        if float((b - a) @ vcur) <= 0: continue
        u, dist = closest_on_seg(a, b, p)
        if best is None or dist < best[0]:
            best = (dist, k, sa + u * (sb - sa))
    if best is None or best[0] > 0.02: return None
    _, k0, s_match = best
    # walk forward to closest approach to T
    arr = None
    for k in range(k0, len(hist) - 1):
        (sa, a), (sb, b) = hist[k], hist[k + 1]
        if sa > s_match + 1.2 * period: break
        if sb - sa > max_gap: continue
        u, dist = closest_on_seg(a, b, T)
        t = sa + u * (sb - sa)
        if t < s_match: continue
        if arr is None or dist < arr[0] - 1e-9:
            arr = (dist, t)
        elif dist > arr[0] + 0.03:
            break
    if arr is None: return None
    return arr[1] - s_match, arr[0]
