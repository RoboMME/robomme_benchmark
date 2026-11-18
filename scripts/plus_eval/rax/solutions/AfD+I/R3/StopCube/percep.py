import numpy as np
from scipy import ndimage
from rax import backproject

def nontable_mask(rgb):
    im = rgb.astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    table = (R - G > 45) & (G - B > 15)
    m = ~table
    return m

def components(meta, cam="front", min_px=8, ymin=None):
    rgb = meta["_imgs"][(cam, "rgb")]; dep = meta["_imgs"][(cam, "depth")]
    m = nontable_mask(rgb)
    pts_all = None
    vs, us = np.nonzero(np.ones_like(m))
    lab, n = ndimage.label(m)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < min_px: continue
        d = dep[ys, xs]; ok = d > 0
        if ok.sum() < 3: continue
        P = backproject(meta, cam, xs[ok], ys[ok], d[ok])
        col = rgb[ys, xs].reshape(-1, 3).mean(0)
        out.append(dict(n=len(xs), u=xs.mean(), v=ys.mean(), xyz=np.median(P, 0), zmax=np.percentile(P[:, 2], 95),
                        zmin=np.percentile(P[:,2],5), color=col, bbox=(xs.min(), ys.min(), xs.max(), ys.max())))
    return out, lab

def height_map(meta, cam="front"):
    dep = meta["_imgs"][(cam, "depth")]
    vs, us = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
    P = backproject(meta, cam, us.ravel(), vs.ravel(), dep.ravel()).reshape(dep.shape[0], dep.shape[1], 3)
    P[dep <= 0] = np.nan
    return P

def cube_obs(P, rgb, button_xy, zlo=0.022, zhi=0.045, excl=0.07, vmin=28):
    z = P[..., 2]
    m = (z > zlo) & (z < zhi) & nontable_mask(rgb)
    m[:vmin] = False
    d = np.hypot(P[..., 0] - button_xy[0], P[..., 1] - button_xy[1])
    m &= ~(d < excl)
    lab, n = ndimage.label(m)
    best = None
    for k in range(1, n + 1):
        sel = lab == k
        c = int(sel.sum())
        if c < 4: continue
        if best is None or c > best[0]:
            best = (c, sel)
    if best is None:
        return None
    sel = best[1]
    xy = np.array([np.median(P[..., 0][sel]), np.median(P[..., 1][sel])])
    ys, xs = np.nonzero(sel)
    return dict(xy=xy, n=best[0], u=xs.mean(), v=ys.mean())
