"""Depth-based perception for StopCube (front camera)."""
import numpy as np
from scipy import ndimage
import pkt

CUBE_ZMIN, CUBE_ZMAX = 0.022, 0.07
FLAT_ZMIN, FLAT_ZMAX = 0.008, 0.021


def blobs(mask, P, min_px=4):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_px:
            continue
        pts = P[ys, xs]
        out.append(dict(n=len(ys), u=float(xs.mean()), v=float(ys.mean()),
                        xyz=np.nanmedian(pts, 0), zmax=float(np.nanmax(pts[:, 2])), ys=ys, xs=xs))
    return out


def frame_geom(meta, imgs):
    P = pkt.backproject(meta, imgs[('front', 'depth')], 'front')
    return P


def find_high(P, exclude=(), excl_r=0.06):
    z = P[..., 2]
    mask = (z > CUBE_ZMIN) & (z < CUBE_ZMAX)
    for c in exclude:
        d = np.hypot(P[..., 0] - c[0], P[..., 1] - c[1])
        mask &= ~(d < excl_r)
    return blobs(mask, P)


def find_flat(P):
    z = P[..., 2]
    return (z > FLAT_ZMIN) & (z < FLAT_ZMAX)
