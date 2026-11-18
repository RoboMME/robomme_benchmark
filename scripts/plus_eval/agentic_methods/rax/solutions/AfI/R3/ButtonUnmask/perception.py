import numpy as np
from scipy import ndimage

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def deproject(depth_m, K, pose):
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    xc = (u-cx)/fx*depth_m; yc = -(v-cy)/fy*depth_m; zc = -depth_m
    P = np.stack([xc, yc, zc], -1)
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    P = P @ R.T + t
    P[depth_m <= 0] = np.nan
    return P

def cloud(m, A, cam):
    return deproject(A[(cam, "depth")], m["camera_intrinsics"][cam]["matrix"], m["camera_poses"][cam])

def white_mask(rgb):
    rgb = rgb.astype(int)
    return (rgb.min(2) > 140) & (rgb.max(2) - rgb.min(2) < 45)

def find_bins(m, A, cam="front", zlo=0.045, zhi=0.095, min_px=60):
    """White blobs whose top lies at bin height. Returns list of dicts with xy, ztop, yaw, ext."""
    P = cloud(m, A, cam)
    z = P[..., 2]
    w = white_mask(A[(cam, "rgb")]) & (z > zlo) & (z < zhi)
    lab, n = ndimage.label(w)
    out = []
    for i in range(1, n+1):
        msk = lab == i
        if msk.sum() < min_px: continue
        pts = P[msk]
        r = rect_fit(pts)
        if r is None: continue
        ys, xs = np.where(msk)
        r.update(npx=int(msk.sum()), uv=(float(xs.mean()), float(ys.mean())))
        out.append(r)
    return out

def rect_fit(pts):
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts) < 10: return None
    zt = np.percentile(pts[:, 2], 98)
    top = pts[pts[:, 2] > zt-0.015][:, :2]
    best = None
    for a in np.deg2rad(np.arange(-45, 45, 1.0)):
        c, s = np.cos(a), np.sin(a); R = np.array([[c, s], [-s, c]])
        q = top @ R.T; ext = q.max(0)-q.min(0); area = ext[0]*ext[1]
        if best is None or area < best[0]:
            ctr = (q.max(0)+q.min(0))/2; best = (area, a, ext, ctr @ R)
    return dict(xy=best[3].tolist(), yaw=float(best[1]), ext=best[2].tolist(), ztop=float(zt), n=int(len(top)))

def colored_points(m, A, cam, color, zlo=0.0, zhi=0.08):
    """Return mask of pixels of a named saturated color."""
    rgb = A[(cam, "rgb")].astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(2); mn = rgb.min(2); sat = mx - mn
    if color == "red": msk = (r > 120) & (g < 0.35*r) & (b < 0.35*r)
    elif color == "green": msk = (g > 120) & (r < 0.5*g) & (b < 0.5*g)
    elif color == "blue": msk = (b > 120) & (r < 0.5*b) & (g < 0.5*b)
    else: msk = np.zeros(r.shape, bool)
    return msk & (sat > 60)
