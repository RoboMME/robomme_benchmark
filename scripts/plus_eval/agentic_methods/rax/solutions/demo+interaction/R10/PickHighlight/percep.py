import numpy as np
from PIL import Image

def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(depth_m, K, pose):
    """depth_m HxW meters -> HxWx3 world points."""
    H, W = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    v, u = np.mgrid[0:H, 0:W].astype(float)
    d = depth_m
    cam = np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], -1)
    R = quat_to_R(pose["quaternion_wxyz"])
    return cam @ R.T + np.array(pose["position_m"])

def load(path):
    return np.asarray(Image.open(path))

def green_mask(rgb):
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > r + 25) & (g > 90) & (g >= b - 10)

def white_mask(rgb):
    a = rgb.astype(int)
    return (a.min(-1) > 200) & ((a.max(-1) - a.min(-1)) < 30)

def components(mask, min_px=8):
    H, W = mask.shape
    lab = np.zeros((H, W), int); n = 0; comps = []
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and lab[y0, x0] == 0:
                n += 1; stack = [(y0, x0)]; lab[y0, x0] = n; pts = []
                while stack:
                    y, x = stack.pop(); pts.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                            lab[yy, xx] = n; stack.append((yy, xx))
                if len(pts) >= min_px:
                    comps.append(np.array(pts))
    return comps

def find_cubes(rgb, pts_world, table_z=None):
    """Return list of dicts: pixel centroid (u,v), world xyz of top, npx."""
    gm = green_mask(rgb)
    out = []
    for c in components(gm, 6):
        vs, us = c[:, 0], c[:, 1]
        P = pts_world[vs, us]
        out.append({"u": float(us.mean()), "v": float(vs.mean()), "npx": len(c),
                    "xyz": np.median(P, 0).tolist(), "zmax": float(np.percentile(P[:, 2], 90))})
    return out

def ring_score(cube, new_white, r_in=3, r_out=16):
    H, W = new_white.shape
    v, u = np.mgrid[0:H, 0:W]
    d = np.hypot(u - cube["u"], v - cube["v"])
    ann = (d >= r_in) & (d <= r_out)
    return float(new_white[ann].mean())

def find_button(rgb, pts_world):
    """White square with grey disc: largest white component present at baseline."""
    wm = white_mask(rgb) & (pts_world[..., 2] < 0.15)
    comps = components(wm, 10)
    if not comps:
        return None
    best = None
    for c in comps:
        vs, us = c[:, 0], c[:, 1]
        # grey pixels inside bbox
        sub = rgb[vs.min():vs.max()+1, us.min():us.max()+1].astype(int)
        grey = ((sub.max(-1) - sub.min(-1)) < 20) & (sub.mean(-1) > 90) & (sub.mean(-1) < 180)
        score = len(c) + 5 * grey.sum()
        if best is None or score > best[0]:
            best = (score, c, grey.sum())
    c = best[1]
    vs, us = c[:, 0], c[:, 1]
    uc, vc = us.mean(), vs.mean()
    P = pts_world[vs, us]
    return {"u": float(uc), "v": float(vc), "xyz": np.median(P, 0).tolist(), "npx": len(c), "grey": int(best[2])}

def obs_arrays(fr_or_packet_files, cam, meta):
    pass

def raised_mask(P, zlo=0.025, zhi=0.08):
    return (P[..., 2] > zlo) & (P[..., 2] < zhi)

def cluster_world(P, mask, link=0.012, min_pts=15):
    """Cluster masked world points by xy proximity (grid BFS on pixel neighbours)."""
    comps = components(mask, min_pts)
    out = []
    for c in comps:
        pts = P[c[:, 0], c[:, 1]]
        out.append({"pix": c, "xyz": pts.mean(0), "n": len(c), "u": float(c[:, 1].mean()), "v": float(c[:, 0].mean()),
                    "zmax": float(np.percentile(pts[:, 2], 95))})
    return out
