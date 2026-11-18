import numpy as np
from perc import parse, backproject

COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]

def hsv(rgb):
    a = rgb[..., :3].astype(np.float64) / 255.0
    mx = a.max(-1); mn = a.min(-1); d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = d > 1e-6
    i = m & (mx == r); h[i] = ((g - b)[i] / d[i]) % 6
    i = m & (mx == g) & (mx != r); h[i] = (b - r)[i] / d[i] + 2
    i = m & (mx == b) & (mx != r) & (mx != g); h[i] = (r - g)[i] / d[i] + 4
    h = h * 60
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx

def color_mask(rgb, color):
    h, s, v = hsv(rgb)
    sat = (s > 0.6) & (v > 0.25)
    if color == "red": return sat & ((h < 12) | (h > 345))
    if color == "green": return sat & (h > 90) & (h < 160)
    if color == "blue": return sat & (h > 200) & (h < 260)
    if color == "yellow": return sat & (h > 45) & (h < 70)
    if color == "orange": return sat & (h > 15) & (h < 40)
    if color == "purple": return sat & (h > 265) & (h < 320)
    return sat

def components(mask, min_px=4):
    from collections import deque
    H, W = mask.shape; seen = np.zeros_like(mask, bool); out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pix = []
        while q:
            y, x = q.popleft(); pix.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(pix) >= min_px: out.append(np.array(pix))
    return out

def blobs(rgb, pw, mask, min_px=4):
    res = []
    for pix in components(mask, min_px):
        pts = pw[pix[:, 0], pix[:, 1]]
        pts = pts[~np.isnan(pts).any(1)]
        if len(pts) == 0: continue
        res.append(dict(n=len(pix), uv=(float(pix[:, 1].mean()), float(pix[:, 0].mean())),
                        xyz=np.median(pts, 0), zmax=float(np.percentile(pts[:, 2], 95)),
                        zmin=float(np.percentile(pts[:, 2], 5))))
    return sorted(res, key=lambda b: -b["n"])

def white_mask(rgb):
    h, s, v = hsv(rgb)
    return (s < 0.18) & (v > 0.55)

def analyze(meta, imgs, cube_color):
    rgb = imgs[("front", "rgb")]; pw = backproject(meta, imgs, "front")
    # table plane estimate: median z of lower image
    z = pw[150:250, 20:236, 2]; table_z = float(np.nanmedian(z))
    cube = blobs(rgb, pw, color_mask(rgb, cube_color))
    wm = white_mask(rgb)
    # exclude robot: points far above table or beyond workspace
    ok = (pw[..., 2] < table_z + 0.06) & (pw[..., 2] > table_z - 0.02)
    wb = blobs(rgb, pw, wm & np.nan_to_num(ok).astype(bool), min_px=6)
    # also gray (button cap / target rings): low saturation medium value
    return dict(table_z=table_z, cube=cube, white=wb, pw=pw)
