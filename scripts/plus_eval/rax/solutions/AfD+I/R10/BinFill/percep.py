"""Color-blob perception of cubes, bin and button from RGB-D."""
import numpy as np
from rio import pointcloud

def color_masks(rgb):
    a = rgb.astype(np.int32); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b); mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    m = {}
    m["red"] = (r > 70) & (r > 2.0*g) & (r > 2.0*b) & (g < 60)
    m["green"] = (g > 70) & (g > 1.6*r) & (g > 1.6*b)
    m["blue"] = (b > 60) & (b > 1.8*r) & (b > 1.4*g) & (g < 0.7*b)
    return m

def blobs(mask, min_px=4):
    """Connected components (4-neigh) -> list of pixel index arrays."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32); out = []
    ys, xs = np.nonzero(mask)
    cur = 0
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1; stack = [(y0, x0)]; lab[y0, x0] = cur; pix = []
        while stack:
            y, x = stack.pop(); pix.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur; stack.append((yy, xx))
        if len(pix) >= min_px:
            out.append(np.array(pix))
    return out

def find_cubes(meta, imgs, cam="front", colors=("red", "green", "blue"), min_px=4):
    rgb = imgs[(cam, "rgb")]
    P = pointcloud(meta, imgs, cam)
    ms = color_masks(rgb)
    res = []
    for c in colors:
        for pix in blobs(ms[c], min_px):
            pts = P[pix[:, 0], pix[:, 1]]
            pts = pts[~np.isnan(pts[:, 0])]
            if len(pts) < 3:
                continue
            top = np.percentile(pts[:, 2], 80)
            ctr = np.median(pts, axis=0)
            if top < 0.02 or top > 0.11 or ctr[0] < -0.5:
                continue  # wood speckle, carried cube or robot
            res.append(dict(color=c, xyz=ctr, ztop=float(top), npx=len(pix),
                            uv=(float(pix[:, 1].mean()), float(pix[:, 0].mean()))))
    return res

def find_bin(meta, imgs, cam="front"):
    rgb = imgs[(cam, "rgb")].astype(np.int32)
    P = pointcloud(meta, imgs, cam)
    dark = rgb.max(-1) < 25
    ok = dark & ~np.isnan(P[..., 0]) & (P[..., 0] > -0.5) & (P[..., 2] > -0.02) & (P[..., 2] < 0.1)
    bl = blobs(ok, 20)
    if not bl:
        return None
    def zmed(pix):
        return float(np.median(P[pix[:, 0], pix[:, 1], 2]))
    raised = [b for b in bl if zmed(b) > 0.01]
    pix = max(raised or bl, key=len)
    pts = P[pix[:, 0], pix[:, 1]]
    return dict(xyz=np.median(pts, 0), npx=len(pix), lo=pts.min(0), hi=pts.max(0))

def find_button(meta, imgs, cam="front", bin_xy=None):
    rgb = imgs[(cam, "rgb")].astype(np.int32)
    P = pointcloud(meta, imgs, cam)
    sat = rgb.max(-1) - rgb.min(-1)
    m = (sat < 12) & (rgb.max(-1) > 45) & ~np.isnan(P[..., 0])
    m &= (P[..., 0] > -0.48) & (P[..., 2] > 0.02) & (P[..., 2] < 0.07)
    if bin_xy is not None:
        dxy = np.hypot(P[..., 0]-bin_xy[0], P[..., 1]-bin_xy[1])
        m &= ~(dxy < 0.075)
    bl = blobs(m, 10)
    if not bl:
        return None
    pix = max(bl, key=len)
    pts = P[pix[:, 0], pix[:, 1]]
    return dict(xyz=np.median(pts, 0), ztop=float(np.percentile(pts[:, 2], 90)), npx=len(pix))
