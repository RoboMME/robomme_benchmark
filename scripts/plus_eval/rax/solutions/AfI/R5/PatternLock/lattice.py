"""Fit a 5x5 (possibly rotated/scaled) button lattice to back-projected button points."""
import numpy as np
from geom import button_points


def _lines(vals, gap):
    order = np.argsort(vals)
    v = vals[order]
    groups = [[v[0]]]
    for x in v[1:]:
        if x - groups[-1][-1] > gap:
            groups.append([x])
        else:
            groups[-1].append(x)
    return [(float(np.mean(g)), len(g)) for g in groups]


def _best5(cl, s):
    cl = sorted([c for c in cl if c[1] >= 2])
    best, bscore = None, None
    for a in range(0, len(cl) - 4):
        sub = cl[a:a + 5]
        pos = np.array([c[0] for c in sub])
        gaps = np.diff(pos)
        if gaps.min() < 0.6 * s or gaps.max() > 1.4 * s:
            continue
        score = np.std(gaps) / np.mean(gaps) - 0.02 * sum(c[1] for c in sub)
        if bscore is None or score < bscore:
            best, bscore = pos, score
    return best


def fit_lattice(pts):
    """pts: Nx3 button top points. Returns nodes[5,5,2] (i along ~+x, j along ~+y) or None."""
    if pts is None or len(pts) < 12:
        return None
    P = np.asarray(pts)[:, :2]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    s = float(np.median(nn))
    if not (0.03 < s < 0.2):
        return None
    keep = (D < 1.3 * s).sum(1) >= 2
    P = P[keep]
    if len(P) < 12:
        return None
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, np.inf)
    ang = []
    for a in range(len(P)):
        for b in np.nonzero(D[a] < 1.3 * s)[0]:
            d = P[b] - P[a]
            ang.append(np.arctan2(d[1], d[0]))
    ang = np.array(ang)
    th = np.angle(np.mean(np.exp(4j * ang))) / 4.0  # in (-45deg, 45deg]
    c, si = np.cos(th), np.sin(th)
    Rm = np.array([[c, -si], [si, c]])
    Q = P @ Rm  # rotate by -th: coords in lattice frame
    xs = _best5(_lines(Q[:, 0], 0.4 * s), s)
    ys = _best5(_lines(Q[:, 1], 0.4 * s), s)
    if xs is None or ys is None:
        return None
    idx = np.arange(5) - 2
    xs = np.polyval(np.polyfit(idx, xs, 1), idx)
    ys = np.polyval(np.polyfit(idx, ys, 1), idx)
    nodes = np.zeros((5, 5, 2))
    for i in range(5):
        for j in range(5):
            nodes[i, j] = Rm @ np.array([xs[i], ys[j]])
    # residual check: each node should have a detected button nearby (most of them)
    dmin = np.array([np.min(np.linalg.norm(P - nodes[i, j], axis=1)) for i in range(5) for j in range(5)])
    if np.mean(dmin < 0.3 * s) < 0.8:
        return None
    return nodes


def lattice_from_frame(rgb, dep_m, K, pose):
    return fit_lattice(button_points(rgb, dep_m, K, pose))


def affine_between(src_nodes, dst_nodes):
    """Least-squares 2D affine mapping src lattice nodes to dst nodes: returns (A[2,2], t[2])."""
    S = src_nodes.reshape(-1, 2)
    T = dst_nodes.reshape(-1, 2)
    X = np.hstack([S, np.ones((len(S), 1))])
    M, *_ = np.linalg.lstsq(X, T, rcond=None)
    return M[:2].T, M[2]
