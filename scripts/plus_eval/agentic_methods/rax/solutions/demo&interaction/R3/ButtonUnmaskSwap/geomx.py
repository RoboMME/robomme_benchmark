import numpy as np
def rect_fit(cells):
    """Min-area rectangle of XY cells: (yaw_rad in [0,pi/2), side_a, side_b, centre)."""
    best = None
    c0 = cells.mean(0)
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        e1 = np.array([np.cos(a), np.sin(a)]); e2 = np.array([-np.sin(a), np.cos(a)])
        p1, p2 = (cells - c0) @ e1, (cells - c0) @ e2
        ext1 = np.percentile(p1, 98) - np.percentile(p1, 2); ext2 = np.percentile(p2, 98) - np.percentile(p2, 2)
        area = ext1 * ext2
        if best is None or area < best[0]:
            mid = c0 + e1 * (np.percentile(p1, 98) + np.percentile(p1, 2)) / 2 + e2 * (np.percentile(p2, 98) + np.percentile(p2, 2)) / 2
            best = (area, a, ext1, ext2, mid)
    return best[1], best[2], best[3], best[4]
