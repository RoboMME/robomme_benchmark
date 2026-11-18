"""connected components / dilation; scipy if available, else a numpy fallback."""
import numpy as np
try:
    from scipy.ndimage import label, binary_dilation
except Exception:  # pragma: no cover
    def label(mask):
        m = np.asarray(mask, bool); H, W = m.shape
        lab = np.zeros((H, W), np.int32); cur = 0
        ys, xs = np.nonzero(m)
        for y0, x0 in zip(ys, xs):
            if lab[y0, x0]: continue
            cur += 1; stack = [(y0, x0)]; lab[y0, x0] = cur
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = cur; stack.append((yy, xx))
        return lab, cur
    def binary_dilation(m, iterations=1):
        m = np.asarray(m, bool).copy()
        for _ in range(iterations):
            o = m.copy()
            o[1:] |= m[:-1]; o[:-1] |= m[1:]; o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
            m = o
        return m
