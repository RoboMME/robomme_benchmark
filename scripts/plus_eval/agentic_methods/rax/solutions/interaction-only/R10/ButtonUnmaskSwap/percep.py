import json, math
import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def depth_to_world(depth_m, K, pose):
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam @ R.T + np.array(pose["position_m"])
    return pts

def load_depth(path, mpu=0.001):
    a = np.array(Image.open(path)).astype(np.float64) * mpu
    return a

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def label(mask):
    """simple connected components (4-neigh) without scipy"""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except Exception:
        pass
    H, W = mask.shape
    lab = np.zeros((H, W), int); n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                n += 1; st = [(i, j)]; lab[i, j] = n
                while st:
                    a, b = st.pop()
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        p, q = a+da, b+db
                        if 0 <= p < H and 0 <= q < W and mask[p, q] and lab[p, q] == 0:
                            lab[p, q] = n; st.append((p, q))
    return lab, n

def blobs_from_points(P, valid, zlo=0.045, zhi=0.14, xlim=(-0.30, 0.30), ylim=(-0.40, 0.40), cell=0.005, min_cells=6):
    """Rasterize elevated world points top-down; return list of dict(xy, n, ztop)."""
    from scipy import ndimage
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    m = valid & (z > zlo) & (z < zhi) & (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
    xs, ys, zs = x[m], y[m], z[m]
    nx = int((xlim[1]-xlim[0])/cell); ny = int((ylim[1]-ylim[0])/cell)
    ix = ((xs-xlim[0])/cell).astype(int).clip(0, nx-1); iy = ((ys-ylim[0])/cell).astype(int).clip(0, ny-1)
    grid = np.zeros((nx, ny), int); np.add.at(grid, (ix, iy), 1)
    occ = grid > 0
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, n = ndimage.label(occ)
    out = []
    for k in range(1, n+1):
        cells = np.argwhere(lab == k)
        if len(cells) < min_cells: continue
        cx = xlim[0] + (cells[:, 0].mean()+0.5)*cell; cy = ylim[0] + (cells[:, 1].mean()+0.5)*cell
        sel = lab[ix, iy] == k
        out.append(dict(xy=(float(cx), float(cy)), n=int(len(cells)), ztop=float(np.percentile(zs[sel], 95)) if sel.any() else 0.0))
    return out

COLOR_NAMES = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "cyan", "brown", "gray"]

def classify_rgb(rgb):
    """rgb: Nx3 uint8 -> list of color names (coarse)."""
    import colorsys
    out = []
    for r, g, b in rgb.astype(float) / 255.0:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h *= 360
        if v < 0.22:
            out.append("black"); continue
        if s < 0.25:
            out.append("white" if v > 0.6 else "gray"); continue
        if h < 15 or h >= 340: out.append("red")
        elif h < 50: out.append("orange")
        elif h < 75: out.append("yellow")
        elif h < 165: out.append("green")
        elif h < 200: out.append("cyan")
        elif h < 260: out.append("blue")
        elif h < 340: out.append("purple")
    return out

def detect_cubes(rgb, P, valid, zlo=0.008, zhi=0.06, xlim=(-0.45, 0.30), ylim=(-0.40, 0.40)):
    """Find small elevated objects (cubes) on the table; return list of dict(xy, color, n)."""
    from scipy import ndimage
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    m = valid & (z > zlo) & (z < zhi) & (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
    lab, n = ndimage.label(m)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        cnt = int(sel.sum())
        if cnt < 8:
            continue
        pts = P[sel]
        names = classify_rgb(rgb[sel])
        from collections import Counter
        c = Counter(names).most_common()
        out.append(dict(xy=(float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))), color=c[0][0],
                        votes=dict(c), n=cnt, ztop=float(np.percentile(pts[:, 2], 95)),
                        uv=(float(np.argwhere(sel)[:, 1].mean()), float(np.argwhere(sel)[:, 0].mean()))))
    return out

def raster_blobs(point_sets, zlo=0.045, zhi=0.085, xlim=(-0.45, 0.30), ylim=(-0.40, 0.40), cell=0.005,
                 min_cells=6, static_mask=None, exclude=None):
    """point_sets: list of (P[...,3], valid). Fuse into one top-down occupancy grid; return blobs.
    static_mask: bool grid to remove; exclude: list of (xy, radius) discs to remove."""
    from scipy import ndimage
    nx = int(round((xlim[1]-xlim[0])/cell)); ny = int(round((ylim[1]-ylim[0])/cell))
    grid = np.zeros((nx, ny), int)
    for P, valid in point_sets:
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        m = valid & (z > zlo) & (z < zhi) & (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
        ix = ((x[m]-xlim[0])/cell).astype(int).clip(0, nx-1); iy = ((y[m]-ylim[0])/cell).astype(int).clip(0, ny-1)
        np.add.at(grid, (ix, iy), 1)
    occ = grid > 0
    if static_mask is not None:
        occ &= ~static_mask
    if exclude:
        gx = xlim[0] + (np.arange(nx)+0.5)*cell; gy = ylim[0] + (np.arange(ny)+0.5)*cell
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        for (cx, cy), rad in exclude:
            occ &= ((GX-cx)**2 + (GY-cy)**2) > rad*rad
    occ = ndimage.binary_closing(occ, iterations=1)
    lab, n = ndimage.label(occ)
    out = []
    for k in range(1, n+1):
        cells = np.argwhere(lab == k)
        if len(cells) < min_cells: continue
        cx = xlim[0] + (cells[:, 0].mean()+0.5)*cell; cy = ylim[0] + (cells[:, 1].mean()+0.5)*cell
        out.append(dict(xy=(float(cx), float(cy)), n=int(len(cells))))
    return out, occ

def static_mask_from(point_sets, dilate=3, **kw):
    from scipy import ndimage
    _, occ = raster_blobs(point_sets, min_cells=1, **kw)
    return ndimage.binary_dilation(occ, iterations=dilate)


def square_yaw(pts_xy):
    """yaw (rad, in [-pi/4, pi/4)) of a square-ish point set via min-area bounding box"""
    c = pts_xy - pts_xy.mean(0)
    best = (1e9, 0.0)
    for a in np.deg2rad(np.arange(-45, 45, 2.0)):
        ca, sa = np.cos(a), np.sin(a)
        q = c @ np.array([[ca, -sa], [sa, ca]])
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if area < best[0]:
            best = (area, a)
    return best[1]
