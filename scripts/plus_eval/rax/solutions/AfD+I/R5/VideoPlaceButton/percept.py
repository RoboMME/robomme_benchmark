import numpy as np
from robot import backproject
from video import cube_mask, blobs, target_mask

def cube_world(meta, imgs, color, cam='front'):
    rgb = imgs[(cam, 'rgb')]; dep = imgs[(cam, 'depth')]
    mk = cube_mask(rgb, color)
    vs, us = np.nonzero(mk)
    if len(us) < 5:
        return None
    # keep largest blob
    from scipy import ndimage
    lab, n = ndimage.label(mk)
    sizes = ndimage.sum(mk, lab, np.arange(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    vs, us = np.nonzero(lab == k)
    pw, ok = backproject(meta, dep, cam, us, vs)
    pw = pw[ok]
    if len(pw) < 5:
        return None
    zt = np.percentile(pw[:, 2], 90)
    top = pw[pw[:, 2] > zt - 0.008]
    c = top.mean(0)
    # yaw estimate from top points (PCA not reliable for square); use min-area rect search
    xy = top[:, :2] - c[:2]
    best = None
    for a in np.linspace(0, np.pi / 2, 46, endpoint=False):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = xy @ R.T
        ext = (q.max(0) - q.min(0)).prod()
        if best is None or ext < best[0]:
            best = (ext, a)
    return dict(center=c, top_z=float(zt), yaw=float(best[1]), n=len(pw), px=(float(us.mean()), float(vs.mean())))

def target_world_near(meta, imgs, px, cam='front', maxd=25):
    rgb = imgs[(cam, 'rgb')]; dep = imgs[(cam, 'depth')]
    from scipy import ndimage
    mk = target_mask(rgb)
    mk2 = ndimage.binary_closing(mk, iterations=1)
    lab, n = ndimage.label(mk2)
    best = None
    for i in range(1, n + 1):
        vs, us = np.nonzero(lab == i)
        if len(us) < 15:
            continue
        d = np.hypot(us.mean() - px[0], vs.mean() - px[1])
        if best is None or d < best[0]:
            best = (d, us, vs)
    if best is None or best[0] > maxd:
        us = np.array([int(round(px[0]))]); vs = np.array([int(round(px[1]))])
        dd = None
    else:
        dd = best[0]; us, vs = best[1], best[2]
    pw, ok = backproject(meta, dep, cam, us, vs)
    pw = pw[ok]
    if len(pw) == 0:
        return None
    c = np.median(pw, 0)
    return dict(center=c, pix_dist=dd, n=len(pw))
