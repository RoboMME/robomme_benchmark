import numpy as np
from scipy import ndimage

def background(frames):
    """frames: list of HxWx3 uint8 arrays. Per-pixel median."""
    stack = np.stack(frames[:: max(1, len(frames)//120)]).astype(np.int16)
    return np.median(stack, axis=0).astype(np.int16)

def fg_mask(img, bg, thr=45, sat_thr=35):
    im = img.astype(np.int16)
    diff = np.abs(im - bg).max(axis=2)
    sat = im.max(axis=2) - im.min(axis=2)
    m = (diff > thr) & (sat > sat_thr)
    m = ndimage.binary_opening(m, iterations=1)
    return m

def blobs(mask, minarea=6):
    lab, n = ndimage.label(mask)
    out = []
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(mask, lab, idx)
    coms = ndimage.center_of_mass(mask, lab, idx)
    for a, (y, x) in zip(areas, coms):
        if a >= minarea:
            out.append((float(x), float(y), int(a)))
    return out
