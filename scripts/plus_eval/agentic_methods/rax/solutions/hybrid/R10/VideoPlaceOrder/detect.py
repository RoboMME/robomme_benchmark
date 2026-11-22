import numpy as np
from scipy import ndimage
def purple_mask(a):
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 140) & (b > 150) & (g < 130) & (b - g > 60)
def color_mask(a, col):
    a = a.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if col == 'red': return (r > 150) & (g < 70) & (b < 70)
    if col == 'green': return (g > 140) & (r < 100) & (b < 100)
    if col == 'blue': return (b > 140) & (r < 70) & (g < 70)
    raise ValueError(col)
def blobs(mask, min_px=5, dil=2):
    lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=dil) if dil else mask)
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero((lab == k) & mask)
        if len(xs) >= min_px:
            out.append((xs.mean(), ys.mean(), len(xs), xs.min(), xs.max(), ys.min(), ys.max()))
    return out
