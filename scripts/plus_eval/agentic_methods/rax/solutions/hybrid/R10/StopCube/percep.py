import numpy as np
from scipy import ndimage
import rlib

def height_map(meta, imgs):
    P = rlib.world_map(meta, imgs, "front")
    return P

def components(mask, min_px=6):
    lab, n = ndimage.label(mask)
    out = []
    for k in range(1, n+1):
        ys, xs = np.nonzero(lab == k)
        if len(xs) >= min_px:
            out.append((ys, xs))
    return out

def high_blobs(P, zlo=0.022, zhi=0.06, min_px=6):
    """Blobs of points between zlo and zhi; returns list of dict(xy, n, z)."""
    z = P[..., 2]
    mask = np.isfinite(z) & (z > zlo) & (z < zhi)
    res = []
    for ys, xs in components(mask, min_px):
        pts = P[ys, xs]
        res.append({"xy": np.nanmedian(pts[:, :2], 0), "n": len(xs), "z": float(np.nanmax(pts[:, 2])),
                    "uv": (float(xs.mean()), float(ys.mean()))})
    return res
