"""Pure numpy/PIL perception for VideoPlaceOrder (front camera RGB)."""
import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None


def label(mask):
    """Connected components (4-connectivity). Returns (labels, n)."""
    if _ndi is not None:
        return _ndi.label(mask)
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    n = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        n += 1
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
    return lab, n


def _shift_or(m):
    n = m.copy()
    n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
    return n


def erode(mask, it=1):
    m = mask.astype(bool)
    for _ in range(it):
        m = ~_shift_or(~m)
    return m


def closing(mask, it=1):
    if _ndi is not None:
        return _ndi.binary_closing(mask, iterations=it)
    # numpy fallback (border treated like scipy: outside = False after erosion)
    m = mask.astype(bool)
    for _ in range(it):
        m = _shift_or(m)
    for _ in range(it):
        m = ~_shift_or(~m)
    return m


def purple_mask(im):
    im = im.astype(np.int32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    return (B - G > -8) & (R - G > 22) & (B > 60) & (R > 90)


CUBE_COLORS = ("red", "green", "blue")


def cube_mask(im, color):
    im = im.astype(np.int32)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    if color == "red":
        return (R > 110) & (G < 0.35 * R) & (B < 0.35 * R) & (R - G > 90)
    if color == "green":
        return (G > 110) & (R < 0.55 * G) & (B < 0.55 * G)
    if color == "blue":
        return (B > 100) & (R < 0.45 * B) & (G < 0.45 * B)
    raise ValueError(color)


def blobs(mask, min_area=4, max_area=100000):
    lab, n = label(mask)
    out = []
    if n == 0:
        return out
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    cnt = np.bincount(ids, minlength=n + 1)
    sx = np.bincount(ids, weights=xs, minlength=n + 1)
    sy = np.bincount(ids, weights=ys, minlength=n + 1)
    x0 = np.full(n + 1, 10 ** 6); x1 = np.full(n + 1, -1)
    y0 = np.full(n + 1, 10 ** 6); y1 = np.full(n + 1, -1)
    np.minimum.at(x0, ids, xs); np.maximum.at(x1, ids, xs)
    np.minimum.at(y0, ids, ys); np.maximum.at(y1, ids, ys)
    for i in range(1, n + 1):
        a = int(cnt[i])
        if a < min_area or a > max_area:
            continue
        bw = int(x1[i] - x0[i] + 1); bh = int(y1[i] - y0[i] + 1)
        out.append(dict(x=float(sx[i] / a), y=float(sy[i] / a), area=a, w=bw, h=bh,
                        fill=a / float(bw * bh)))
    return out


def target_blobs(im):
    m = closing(purple_mask(im), 1)
    bl = blobs(m, min_area=12, max_area=900)
    # targets are compact ellipses, wider than tall in the front view
    return [b for b in bl if b["w"] <= 40 and b["h"] <= 30 and b["fill"] > 0.35]


def cube_blobs(im, color):
    m = cube_mask(im, color)
    return blobs(m, min_area=6, max_area=3000)


def dilate(mask, it=1):
    if _ndi is not None:
        return _ndi.binary_dilation(mask, iterations=it)
    m = mask.copy()
    for _ in range(it):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m
