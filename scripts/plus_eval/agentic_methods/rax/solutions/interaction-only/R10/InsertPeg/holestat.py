import numpy as np
from geom import backproject

def hole_stats(depth, K, pose, F, d, lat, tcp):
    """Interior-of-hole points seen through the box face (beyond the face plane along d)."""
    P, valid = backproject(depth, K, pose)
    rel = P - F; s = rel @ d; l = rel @ lat; z = P[..., 2]
    s_min = float((np.asarray(tcp) - F) @ d) + 0.09
    win = valid & (np.abs(l) < 0.04) & (np.abs(z - F[2]) < 0.03) & (s > s_min) & (s < 0.15)
    if win.sum() < 30:
        return None
    sv = s[win]
    hist, edges = np.histogram(sv, bins=np.arange(s_min, 0.15, 0.001))
    j = int(np.argmax(hist)); s_face = 0.5 * (edges[j] + edges[j + 1])
    near = sv[np.abs(sv - s_face) < 0.002]
    s_face = float(np.median(near))
    hole = win & (s > s_face + 0.004)
    n = int(hole.sum())
    if n < 15:
        return dict(s_face=s_face, n=n)
    hl, hz = l[hole], z[hole]
    return dict(s_face=s_face, n=n, floor=float(np.percentile(hz, 5)), l_lo=float(np.percentile(hl, 3)),
                l_hi=float(np.percentile(hl, 97)), l_mid=float(0.5 * (np.percentile(hl, 3) + np.percentile(hl, 97))))
