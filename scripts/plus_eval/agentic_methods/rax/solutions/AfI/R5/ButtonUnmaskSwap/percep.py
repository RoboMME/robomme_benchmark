import numpy as np
from scipy import ndimage

def blobs(o, cam="front", zmin=0.006, zmax=0.2, min_px=4, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6)):
    pw, ok = o.world_map(cam)
    rgb = o.rgb[cam].astype(float)
    m = ok & (pw[..., 2] > zmin) & (pw[..., 2] < zmax) & (pw[..., 0] > xlim[0]) & (pw[..., 0] < xlim[1]) & (pw[..., 1] > ylim[0]) & (pw[..., 1] < ylim[1])
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        mm = lab == i
        k = int(mm.sum())
        if k < min_px:
            continue
        pts = pw[mm]
        col = rgb[mm].mean(0)
        vs, us = np.nonzero(mm)
        out.append(dict(n=k, xyz=pts.mean(0), zmax=float(pts[:, 2].max()), rgb=col, uv=(float(us.mean()), float(vs.mean())),
                        ext=(pts.max(0) - pts.min(0))))
    return out

def color_name(rgb):
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 30:
        return "white" if mx > 150 else ("black" if mx < 70 else "gray")
    h = 0
    if mx == r: h = (60 * (g - b) / (mx - mn)) % 360
    elif mx == g: h = 60 * (b - r) / (mx - mn) + 120
    else: h = 60 * (r - g) / (mx - mn) + 240
    if h < 20 or h >= 330: return "red"
    if h < 45: return "orange"
    if h < 70: return "yellow"
    if h < 170: return "green"
    if h < 260: return "blue"
    return "purple"
