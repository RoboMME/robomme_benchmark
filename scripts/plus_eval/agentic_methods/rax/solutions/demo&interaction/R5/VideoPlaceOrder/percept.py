"""Live perception: cube and target world positions from the front RGB-D view."""
import numpy as np
import video
import robot as rb


def cube_world(meta, imgs, color, cam='front'):
    a = imgs[(cam, 'rgb')]; d = imgs[(cam, 'depth')]
    mk = video.cube_mask(a, color)
    from scipy import ndimage
    lab, n = ndimage.label(mk)
    if n == 0:
        return None
    sizes = ndimage.sum(mk, lab, range(1, n + 1))
    j = int(np.argmax(sizes)) + 1
    if sizes[j - 1] < 6:
        return None
    vs, us = np.nonzero(lab == j)
    P = rb.backproject(meta, d, cam, us, vs)
    P = P[~np.isnan(P).any(1)]
    if len(P) < 3:
        return None
    ztop = np.percentile(P[:, 2], 90)
    top = P[P[:, 2] > ztop - 0.006]
    c = np.median(top, axis=0)
    return {'xy': c[:2], 'ztop': float(ztop), 'npx': int(len(us)), 'uv': (float(us.mean()), float(vs.mean()))}


def pixel_world(meta, imgs, u, v, r=2, cam='front'):
    d = imgs[(cam, 'depth')]
    us, vs = np.meshgrid(np.arange(u - r, u + r + 1), np.arange(v - r, v + r + 1))
    P = rb.backproject(meta, d, cam, us.ravel(), vs.ravel())
    P = P[~np.isnan(P).any(1)]
    return np.median(P, axis=0)
