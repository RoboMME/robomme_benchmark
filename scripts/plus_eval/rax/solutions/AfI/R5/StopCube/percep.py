import numpy as np
from scipy import ndimage

def _ch(rgb):
    return [rgb[..., i].astype(int) for i in range(3)]

def wood_mask(rgb):
    r, g, b = _ch(rgb)
    rf = np.maximum(r, 1).astype(float)
    return (r >= g) & (g >= b) & (g / rf > 0.35) & (g / rf < 0.75) & (b / rf < 0.6)

def cube_mask(rgb):
    """Generic: anything not wood, not neutral gray/white, not target purple."""
    r, g, b = _ch(rgb)
    neutral = (abs(r - g) < 14) & (abs(g - b) < 14)
    purple = target_mask(rgb) | ((b > r - 5) & (b - g > 50))
    dark = (r + g + b) < 90
    return ~wood_mask(rgb) & ~neutral & ~purple & ~dark

def target_mask(rgb):
    r, g, b = _ch(rgb)
    return (b > 180) & (r > 140) & (g < 130) & (b - g > 60)

def gray_mask(rgb, lo=60, hi=175):
    r, g, b = _ch(rgb)
    return (abs(r - g) < 10) & (abs(g - b) < 10) & (r >= lo) & (r < hi)

def white_mask(rgb):
    r, g, b = _ch(rgb)
    return (abs(r - g) < 10) & (abs(g - b) < 10) & (r >= 200)

def centroid(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 3: return None
    return float(xs.mean()), float(ys.mean()), len(xs)

def largest_component(m, min_size=3):
    lab, n = ndimage.label(m)
    if n == 0: return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    i = int(np.argmax(sizes))
    if sizes[i] < min_size: return None
    return lab == (i + 1)

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def pix_to_world(meta, cam, u, v, depth_img, r=1):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    ui, vi = int(round(u)), int(round(v))
    win = depth_img[max(0, vi-r):vi+r+1, max(0, ui-r):ui+r+1].astype(float)
    win = win[win > 0]
    if len(win) == 0: return None
    d = float(np.median(win)) * 0.001
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.array([(u-cx)*d/fx, -(v-cy)*d/fy, -d])
    return quat_to_R(pose["quaternion_wxyz"]) @ pc + np.array(pose["position_m"])

def mask_to_world(meta, cam, m, depth_img):
    """Back-project every valid pixel of mask m; returns Nx3 array."""
    ys, xs = np.nonzero(m)
    d = depth_img[ys, xs].astype(float) * 0.001
    ok = d > 0
    xs, ys, d = xs[ok], ys[ok], d[ok]
    if len(d) == 0: return None
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pc = np.stack([(xs-cx)*d/fx, -(ys-cy)*d/fy, -d], 1)
    return pc @ quat_to_R(pose["quaternion_wxyz"]).T + np.array(pose["position_m"])

def find_button(meta, rgb, dep, top=25):
    """Gray cap component next to white base. Returns (xy, cap_top_z, info) or None."""
    g = gray_mask(rgb); g[:top] = 0
    w = white_mask(rgb); w[:top] = 0
    t = ndimage.binary_dilation(target_mask(rgb), iterations=4)
    lab, n = ndimage.label(g)
    best = None
    for i in range(1, n + 1):
        m = lab == i
        size = int(m.sum())
        if size < 20 or size > 1500: continue
        ys, xs = np.nonzero(m)
        if ys.min() <= top: continue
        if (m & t).any(): continue
        near = ndimage.binary_dilation(m, iterations=4) & w
        score = int(near.sum())
        if score < 5: continue
        if best is None or score + size > best[0]:
            best = (score + size, m)
    if best is None: return None
    pts = mask_to_world(meta, "front", best[1], dep)
    if pts is None or len(pts) < 5: return None
    zmax = np.percentile(pts[:, 2], 95)
    topp = pts[pts[:, 2] > zmax - 0.004]
    return topp[:, :2].mean(0), float(np.median(topp[:, 2])), int(best[1].sum())

def find_target(meta, rgb, dep, top=25):
    m = target_mask(rgb); m[:top] = 0
    m = largest_component(m, 10)
    if m is None: return None
    pts = mask_to_world(meta, "front", m, dep)
    if pts is None: return None
    return pts.mean(0)

def find_cube(meta, rgb, dep, top=25):
    m = cube_mask(rgb); m[:top] = 0
    m = ndimage.binary_opening(m, iterations=1)
    m = largest_component(m, 20)
    if m is None: return None
    c = centroid(m)
    w = pix_to_world(meta, "front", c[0], c[1], dep, r=1)
    return w, c

# ---------------- v3 generic colour-agnostic perception ----------------
def colored_mask(rgb, top=25, opening=True):
    r, g, b = _ch(rgb)
    neutral = (abs(r - g) < 14) & (abs(g - b) < 14)
    dark = (r + g + b) < 90
    m = ~wood_mask(rgb) & ~neutral & ~dark
    m[:top] = 0
    return ndimage.binary_opening(m, iterations=1) if opening else m

def chroma(rgb_pixels):
    p = rgb_pixels.astype(float)
    return p / (p.sum(-1, keepdims=True) + 1.0)

def find_target_generic(rgb, top=25):
    """Bullseye: coloured component with white rings inside its footprint.
    Returns dict(mask=footprint, colored=colored target pixels, chroma=median chroma, score) or None."""
    col = colored_mask(rgb, top)
    w = white_mask(rgb); w[:top] = 0
    lab, n = ndimage.label(col)
    best = None
    for i in range(1, n + 1):
        m = lab == i
        size = int(m.sum())
        if size < 10 or size > 3000: continue
        foot = ndimage.binary_fill_holes(ndimage.binary_closing(m | (ndimage.binary_dilation(m, iterations=2) & w), iterations=2))
        nw = int((foot & w).sum())
        if nw < 8: continue
        score = nw + size
        if best is None or score > best["score"]:
            best = dict(score=score, colored=m, white=int(nw), size=size)
    if best is None: return None
    m = best["colored"]
    foot = ndimage.binary_dilation(ndimage.binary_fill_holes(ndimage.binary_closing(
        m | (ndimage.binary_dilation(m, iterations=2) & w), iterations=2)), iterations=2)
    best["mask"] = foot
    best["chroma"] = np.median(chroma(rgb[m]), 0)
    return best

def find_cube_generic(meta, rgb, dep, tgt, top=25, chroma_thr=0.05):
    col = colored_mask(rgb, top)
    if tgt is not None:
        ys, xs = np.nonzero(col & tgt["mask"])
        if len(xs):
            dc = np.linalg.norm(chroma(rgb[ys, xs]) - tgt["chroma"], axis=1)
            col[ys[dc < chroma_thr], xs[dc < chroma_thr]] = False
        col = ndimage.binary_opening(col, iterations=1)
    m = largest_component(col, 20)
    if m is None: return None
    c = centroid(m)
    w = pix_to_world(meta, "front", c[0], c[1], dep, r=1)
    if w is None: return None
    return w, c

def target_from_static(rgbs, top=25):
    """rgbs: list of front RGB frames spanning several steps. Static coloured pixels
    (present in every frame) with white rings nearby = target."""
    static = None
    for a in rgbs:
        m = colored_mask(a, top, opening=False)
        static = m if static is None else (static & m)
    if static is None: return None
    rgb = rgbs[-1]
    w = white_mask(rgb); w[:top] = 0
    lab, n = ndimage.label(ndimage.binary_closing(static, iterations=1))
    best = None
    for i in range(1, n + 1):
        m = (lab == i) & static
        size = int(m.sum())
        if size < 6: continue
        foot = ndimage.binary_fill_holes(ndimage.binary_closing(
            m | (ndimage.binary_dilation(m, iterations=2) & w), iterations=2))
        nw = int((foot & w).sum())
        score = nw + size
        if nw >= 5 and (best is None or score > best["score"]):
            best = dict(score=score, colored=m, white=nw, size=size, foot=foot)
    if best is None: return None
    best["mask"] = ndimage.binary_dilation(best["foot"], iterations=2)
    best["chroma"] = np.median(chroma(rgb[best["colored"]]), 0)
    return best

def target_position(tgt, frames, chroma_thr=0.05):
    """frames: list of (meta, rgb, dep). Use frame with most target-coloured pixels in footprint."""
    best = None
    for meta, rgb, dep in frames:
        col = colored_mask(rgb, opening=False)
        ys, xs = np.nonzero(col & tgt["mask"])
        if len(xs) == 0: continue
        ok = np.linalg.norm(chroma(rgb[ys, xs]) - tgt["chroma"], axis=1) < chroma_thr
        if best is None or ok.sum() > best[0]:
            m = np.zeros(col.shape, bool); m[ys[ok], xs[ok]] = True
            best = (int(ok.sum()), meta, m, dep)
    if best is None: return None
    pts = mask_to_world(best[1], "front", best[2], best[3])
    return pts.mean(0), best[0]
