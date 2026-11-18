"""World-frame perception: table height, coloured cubes, white containers, buttons."""
import numpy as np
from scipy import ndimage

GRID = 0.004  # m per cell of the XY occupancy grid


def table_z(view, wp=None, ok=None):
    if wp is None:
        wp, ok = view.world_points()
    z = wp[..., 2][ok]
    if z.size == 0:
        return None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.002))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


COLOR_NAMES = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'cyan']


def classify_color(rgb):
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 70:
        return 'black'
    if mx - mn < 40:
        return 'white' if mn > 150 else 'gray'
    # hue
    import colorsys
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h *= 360
    if h < 15 or h >= 340:
        return 'red'
    if h < 45:
        return 'orange'
    if h < 70:
        return 'yellow'
    if h < 165:
        return 'green'
    if h < 200:
        return 'cyan'
    if h < 255:
        return 'blue'
    if h < 300:
        return 'purple'
    return 'pink'


def blobs_xy(points, min_cells=3, grid=GRID, dilate=0):
    """Cluster world points (N,3) in XY by occupancy-grid connected components.
    Returns list of dict(c=xy centroid, n=points, cells, zmax, pts index array)."""
    if len(points) == 0:
        return []
    xy = points[:, :2]
    lo = xy.min(0) - 2 * grid
    ij = np.floor((xy - lo) / grid).astype(int)
    shape = ij.max(0) + 3
    occ = np.zeros(shape, dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    if dilate:
        occ = ndimage.binary_dilation(occ, iterations=dilate)
    lab, n = ndimage.label(occ)
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        idx = np.nonzero(pl == k)[0]
        cells = int((lab == k).sum())
        if cells < min_cells:
            continue
        P = points[idx]
        out.append({'c': P[:, :2].mean(0), 'n': len(idx), 'cells': cells,
                    'zmax': float(P[:, 2].max()), 'idx': idx,
                    'lo': P[:, :2].min(0), 'hi': P[:, :2].max(0)})
    return out


def detect_cubes(views, xmin=-0.47, buttons=()):
    """Cubes visible at step 0 (before containers drop): list of dict(c=xy, color, cells).
    Uses points 0.008<z<0.05 clustered in XY, dropping button-sized blobs (>~200 cells)."""
    pts, cols = [], []
    for v in views.values():
        wp, ok = v.world_points()
        z = wp[..., 2]
        m = ok & (z > 0.012) & (z < 0.05) & (wp[..., 0] > xmin) & (np.abs(wp[..., 1]) < 0.45) & (wp[..., 0] < 0.4)
        for bc in buttons:
            m &= np.linalg.norm(wp[..., :2] - np.asarray(bc)[None, None], axis=-1) > 0.052
        pts.append(wp[m]); cols.append(v.rgb[m])
    P = np.concatenate(pts); C = np.concatenate(cols).astype(float)
    B = blobs_xy(P, min_cells=4, dilate=1)
    out = []
    for b in B:
        if b['cells'] < 60 or b['cells'] > 260:
            continue
        ext = b['hi'] - b['lo']
        if ext.max() > 0.065:
            continue
        col = np.median(C[b['idx']], 0)
        out.append({'c': b['c'], 'color': classify_color(col), 'rgb': col.tolist(), 'cells': b['cells']})
    return out


def detect_buttons(views, xmin=-0.47):
    """Buttons: low wide blobs (top <0.045, extent ~0.07) -> list of xy centres."""
    pts = []
    for v in views.values():
        wp, ok = v.world_points()
        z = wp[..., 2]
        m = ok & (z > 0.008) & (z < 0.045) & (wp[..., 0] > xmin) & (np.abs(wp[..., 1]) < 0.45) & (wp[..., 0] < 0.4)
        pts.append(wp[m])
    P = np.concatenate(pts)
    B = blobs_xy(P, min_cells=4)
    out = []
    for b in B:
        ext = b['hi'] - b['lo']
        if b['cells'] > 200 and 0.055 < ext.min() and ext.max() < 0.11:
            out.append(b['c'])
    return out
