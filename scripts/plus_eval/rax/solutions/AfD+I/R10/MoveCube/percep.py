import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW metres -> HxWx3 world points (nan for invalid)."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    pts = cam @ R.T + np.array(pose["position_m"])
    pts[d <= 0] = np.nan
    return pts

def load_frame(files, root="/workspace"):
    out = {}
    for f in files:
        im = Image.open(f"{root}/{f['path']}")
        a = np.array(im)
        if f["kind"] == "depth":
            a = a.astype(np.float64) * 0.001
        else:
            a = np.array(im.convert("RGB")).astype(np.int32)
        out[(f["camera"], f["kind"])] = a
    return out

def world_points(frame_meta, arrays, cam="front"):
    K = frame_meta["camera_intrinsics"][cam]["matrix"]
    pose = frame_meta["camera_poses"][cam]
    return backproject(arrays[(cam, "depth")], K, pose)

# colour masks on int RGB arrays
def mask_yellow(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 120) & (g > 0.72 * r) & (b < 0.55 * g) & (g > 100)

def mask_blue(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > r + 30) & (b > 90) & (g > r) & (b >= g)

def mask_red(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 120) & (g < 0.3 * r) & (b < 0.3 * r)

def mask_purple(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (b > 120) & (r > 100) & (g < 0.8 * b) & (b > g + 30) & (r > g + 20)

def blobs(mask, min_px=4):
    """connected components (4-neigh) -> list of (n, cu, cv, pixel idx arrays)"""
    H, W = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]; seen[y0, x0] = True; pix = []
        while stack:
            y, x = stack.pop(); pix.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; stack.append((yy, xx))
        if len(pix) >= min_px:
            p = np.array(pix)
            out.append((len(pix), p[:, 1].mean(), p[:, 0].mean(), p))
    out.sort(key=lambda t: -t[0])
    return out

def mask_teal(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (g > 60) & (r < 0.68 * g) & (np.abs(g - b) < 0.12 * g + 6)

def mask_salmon(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 135) & (g > 0.6 * r) & (g < 0.82 * r) & (b > 0.48 * r) & (b < 0.75 * r)

def mask_slate(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 45) & (r < 130) & (g >= r + 3) & (b >= g + 2) & (b < 160) & (b - r < 45)

def mask_tan(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 160) & (g > 0.76 * r) & (g < 0.92 * r) & (b > 0.5 * r) & (b < 0.7 * r)
