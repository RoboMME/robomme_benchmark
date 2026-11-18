"""Turn the demonstrated TCP path into a live-layout path via per-disc-pair similarity maps."""
import numpy as np

def match_discs(Dv, De):
    """Both ordered along line toward +y; returns index map if counts equal else None."""
    if len(Dv) == len(De): return list(range(len(Dv)))
    return None

def _frame(a, b):
    ex = b - a; L = np.linalg.norm(ex); ex = ex / L
    ey = np.array([-ex[1], ex[0]])
    return a, ex, ey, L

def segment_path(xy, D, visit_r=0.02):
    """Return list of (i0, i1, disc_from, disc_to) pieces split at disc visits."""
    d = np.linalg.norm(xy[:, None, :] - D[None], axis=2)  # N x K
    near = d.min(1) < visit_r
    nearest = d.argmin(1)
    # visit events: contiguous runs of near with same disc -> take the index of min distance
    visits = []
    i = 0; N = len(xy)
    while i < N:
        if near[i]:
            k = nearest[i]; j = i
            while j + 1 < N and near[j + 1] and nearest[j + 1] == k: j += 1
            m = i + int(np.argmin(d[i:j + 1, k]))
            visits.append((m, k)); i = j + 1
        else:
            i += 1
    return visits

def map_path(xy, Dv, De, visits):
    """Map video xy path (from the first visit onward) into the live layout."""
    out = []
    K = len(Dv)
    for (m0, k0), (m1, k1) in zip(visits[:-1], visits[1:]):
        if k0 != k1:
            a, b = k0, k1
        else:
            # loop returning to same disc: use neighbour toward the piece centroid
            c = xy[m0:m1 + 1].mean(0)
            cands = [n for n in (k0 - 1, k0 + 1) if 0 <= n < K]
            b = min(cands, key=lambda n: np.linalg.norm(Dv[n] - c)); a = k0
        ov, exv, eyv, Lv = _frame(Dv[a], Dv[b])
        oe, exe, eye, Le = _frame(De[a], De[b])
        seg = xy[m0:m1 + 1] - ov
        u = seg @ exv / Lv; w = seg @ eyv / Lv
        mapped = oe + np.outer(u * Le, exe) + np.outer(w * Le, eye)
        out.append(mapped if not out else mapped[1:])
    return np.vstack(out) if out else np.zeros((0, 2))
