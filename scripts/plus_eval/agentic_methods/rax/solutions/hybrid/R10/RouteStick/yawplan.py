"""Yaw planner for short-stick layouts (v7).
Hand footprint at default yaw relative to the TCP: x in [+0.004, +0.058], y in [-0.095, +0.095].
Obstacles: bar top-face points from front depth. Viterbi over a yaw grid maximises hand-bar
clearance along the low path, with rate limits and a joint window; at target visits the hand may
lift above the bar tops and spin (costly transition)."""
import numpy as np

YAW_WIN = 1.9
BIN = 0.05
MAX_DBIN = 2           # bins per path step
MAX_DP = 0.009
MAX_DYAW = 0.09
SAFE = 0.012
HAND = dict(x0=-0.023, x1=0.023, y=0.104)   # finger plate (only part below bar tops)
BODY = dict(x0=-0.06, x1=0.06, y=0.106)     # thick hand body above the plate (ep9 contact: -x side extends ~6 cm)
BODY_SAFE = 0.006
BODY_W = 3.0
Z_RAMP = 0.002          # m per step for local raises

def clearance_grid(P, yaws, obs, hand=None):
    """(N, J) min distance from obstacle points obs (M,2) to the hand rectangle."""
    hand = HAND if hand is None else hand
    N = len(P); J = len(yaws); out = np.empty((N, J))
    c = np.cos(yaws)[None, :, None]; s = np.sin(yaws)[None, :, None]
    for a in range(0, N, 100):
        p = P[a:a + 100, :2]
        v = obs[None, :, :] - p[:, None, :]          # (n, M, 2)
        vx = v[:, None, :, 0]; vy = v[:, None, :, 1]   # (n,1,M)
        lx = c * vx + s * vy; ly = -s * vx + c * vy    # (n,J,M)
        dx = np.maximum(np.maximum(hand['x0'] - lx, 0), lx - hand['x1'])
        dy = np.maximum(np.abs(ly) - hand['y'], 0)
        out[a:a + 100] = np.hypot(dx, dy).min(2)
    return out

def plan(P, obs, low_mask, lift_ok, z_lift, yaw0=0.0, z_body=None, exempt=None):
    """P (N,3) per-step TCP path; obs (M,2) obstacle xy; low_mask: hand below bar tops;
    lift_ok: indices where a lift-spin is allowed. Returns W, Y (relative yaw), info."""
    N = len(P)
    yaws = np.arange(-YAW_WIN, YAW_WIN + 1e-9, BIN); J = len(yaws)
    clr = clearance_grid(P, yaws, obs)
    cost = np.where(low_mask[:, None], np.maximum(SAFE - clr, 0) / SAFE * 10 + (clr < 0.002) * 100, 0.0)
    if z_body is not None:
        bclr = clearance_grid(P, yaws, obs, BODY)
        body_low = low_mask & (P[:, 2] < z_body)
        cost = cost + np.where(body_low[:, None] & (bclr < BODY_SAFE), BODY_W, 0.0)
    D = np.full((N, J), np.inf); back = np.zeros((N, J), int); lifted = np.zeros((N, J), bool)
    D[0] = cost[0] + 0.2 * np.abs(yaws - yaw0)
    lift_ok = set(int(i) for i in lift_ok)
    for i in range(1, N):
        prev = D[i - 1]
        best = np.full(J, np.inf); arg = np.zeros(J, int)
        for d in range(-MAX_DBIN, MAX_DBIN + 1):
            sh = np.full(J, np.inf)
            if d >= 0: sh[d:] = prev[:J - d] if d else prev
            else: sh[:d] = prev[-d:]
            sh = sh + 0.01 * abs(d)
            m = sh < best; best[m] = sh[m]; arg[m] = (np.arange(J) - d)[m]
        if i in lift_ok:
            j0 = int(np.argmin(prev)); lc = prev[j0] + 3.0 + 0.05 * np.abs(yaws - yaws[j0])
            m = lc < best - 1e-9
            best[m] = lc[m]; arg[m] = j0; lifted[i, m] = True
        D[i] = best + cost[i]; back[i] = arg
    j = int(np.argmin(D[-1])); path = [j]; lifts = []
    for i in range(N - 1, 0, -1):
        if lifted[i, j]: lifts.append(i)
        j = back[i, j]; path.append(j)
    path = path[::-1]; yaw = yaws[path]; lifts = set(lifts)
    n_raise = 0
    if z_body is not None:
        # raise locally (with ramps) wherever the chosen yaw puts the thick body over a bar
        need = body_low & (bclr[np.arange(N), path] < BODY_SAFE)
        if exempt is not None:
            need = need & ~exempt          # stay at the (tested) activation height at target visits
        n_raise = int(need.sum())
        zr = np.where(need, z_body, -np.inf)
        idx = np.nonzero(need)[0]
        if len(idx):
            ii = np.arange(N)
            zlift = np.full(N, -np.inf)
            for j in idx:
                zlift = np.maximum(zlift, z_body - Z_RAMP * np.abs(ii - j))
            P = P.copy(); P[:, 2] = np.maximum(P[:, 2], zlift)
    W = [P[0].copy()]; Y = [yaw0]
    def ramp(p1, y1):
        p0 = W[-1].copy(); y0 = Y[-1]
        n = int(np.ceil(max(np.linalg.norm(p1 - p0) / MAX_DP, abs(y1 - y0) / MAX_DYAW, 1.0)))
        for s in range(1, n + 1):
            W.append(p0 + (p1 - p0) * s / n); Y.append(y0 + (y1 - y0) * s / n)
    ramp(P[0], yaw[0])
    for i in range(1, N):
        if i in lifts:
            base = W[-1].copy(); top = base.copy(); top[2] = max(base[2], z_lift)
            ramp(top, Y[-1]); ramp(top, yaw[i]); ramp(base, yaw[i])
        ramp(P[i], yaw[i])
    pc = clr[np.arange(N), path]
    return np.array(W), np.array(Y), dict(lifts=len(lifts), yaw_range=(float(yaw.min()), float(yaw.max())),
        min_clear=float(pc[low_mask].min()) if low_mask.any() else None,
        n_unsafe=int((pc[low_mask] < SAFE).sum()), n_raise=n_raise, steps=len(W), total=float(D[-1].min()))
