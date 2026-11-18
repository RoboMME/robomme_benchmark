"""Track containers through the conditioning video using front RGB-D projected to world XY."""
import json
import numpy as np
from pathlib import Path
from geom import backproject, load_rgb, load_depth

WS = (-0.50, 0.25, -0.35, 0.35)  # xmin xmax ymin ymax workspace
CELL = 0.004

COLORS = {  # prototype RGB
    "red": (200, 20, 20), "green": (20, 190, 20), "blue": (20, 20, 200),
    "yellow": (220, 210, 20), "orange": (230, 130, 20), "purple": (140, 30, 180),
    "black": (25, 25, 25), "white": (225, 225, 225), "pink": (230, 120, 180),
    "cyan": (20, 200, 200),
}

def classify_color(rgb):
    import colorsys
    r, g, b = [float(v) for v in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 50:
        return "black"
    sat = (mx - mn) / mx
    if sat < 0.25:
        return "white" if mx > 110 else "black"
    h = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0] * 360
    for lim, name in [(15, "red"), (50, "orange"), (75, "yellow"), (165, "green"), (200, "cyan"),
                      (255, "blue"), (290, "purple"), (340, "pink"), (361, "red")]:
        if h < lim:
            return name
    best, bd = None, 1e9
    for k, c in COLORS.items():
        d = (r - c[0]) ** 2 + (g - c[1]) ** 2 + (b - c[2]) ** 2
        if d < bd:
            best, bd = k, d
    return best

def load_frames(manifest_path, root="/workspace"):
    root = Path(root)
    m = json.loads(Path(manifest_path).read_text())
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    out = []
    for f in frames:
        files = {(x["camera"], x["kind"]): x["path"] for x in f["files"]}
        out.append(dict(meta=f, phase=f["phase"], t=f.get("timestamp_s", 0.0),
                        rgb=root / files[("front", "rgb")], depth=root / files[("front", "depth")],
                        wrgb=root / files[("wrist", "rgb")] if ("wrist", "rgb") in files else None,
                        wdepth=root / files[("wrist", "depth")] if ("wrist", "depth") in files else None))
    return out

USE_WRIST = True

def frame_points(fr):
    f = fr["meta"]
    rgb = load_rgb(fr["rgb"])
    d = load_depth(fr["depth"], f["depth"]["front"].get("meters_per_unit", 0.001))
    P = backproject(d, f["camera_intrinsics"]["front"], f["camera_poses"]["front"]).reshape(-1, 3)
    C = rgb.reshape(-1, 3)
    if USE_WRIST and fr.get("wdepth") is not None:
        wd = load_depth(fr["wdepth"], f["depth"]["wrist"].get("meters_per_unit", 0.001))
        wr = load_rgb(fr["wrgb"])
        WP = backproject(wd, f["camera_intrinsics"]["wrist"], f["camera_poses"]["wrist"]).reshape(-1, 3)
        # subsample wrist (close-up, dense) to keep density comparable
        WP, wr = WP.reshape(256, 256, 3)[::2, ::2].reshape(-1, 3), wr[::2, ::2].reshape(-1, 3)
        P = np.concatenate([P, WP]); C = np.concatenate([C, wr])
    return P, C

def in_ws(P):
    return (P[:, 0] > WS[0]) & (P[:, 0] < WS[1]) & (P[:, 1] > WS[2]) & (P[:, 1] < WS[3])

def to_cells(P):
    r = ((P[:, 0] - WS[0]) / CELL).astype(int)
    c = ((P[:, 1] - WS[2]) / CELL).astype(int)
    return r, c

GRID = (int((WS[1] - WS[0]) / CELL) + 1, int((WS[3] - WS[2]) / CELL) + 1)

def components(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    return lab, n

def cells_to_xy(r, c):
    return np.stack([WS[0] + (r + 0.5) * CELL, WS[2] + (c + 0.5) * CELL], -1)

def precompute(frames, log=print):
    """Per frame: top-surface XY points of containers, cube points+colors, tall-static points."""
    data = []
    for fr in frames:
        P, C = frame_points(fr)
        ok = in_ws(P) & np.isfinite(P[:, 2])
        z = P[:, 2]
        top = ok & (z > 0.055) & (z < 0.095)
        low = ok & (z > 0.004) & (z < 0.035)
        tall = ok & (z > 0.10) & (z < 0.16)
        data.append(dict(phase=fr["phase"], top=P[top, :2].astype(np.float32), low=P[low, :2].astype(np.float32),
                         lowc=C[low], tall=P[tall, :2].astype(np.float32),
                         topz=np.median(z[top]) if top.any() else np.nan))
    return data

def blobs(xy, min_size=15):
    from scipy import ndimage
    if len(xy) == 0:
        return []
    g = np.zeros(GRID, bool)
    r, c = to_cells(xy.astype(float))
    g[r, c] = True
    g = ndimage.binary_closing(g, iterations=1)
    lab, n = ndimage.label(g)
    out = []
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) >= min_size:
            out.append(dict(xy=cells_to_xy(rr, cc).mean(0), size=len(rr), cells=(rr, cc)))
    return out

def detect_cubes(data, upto, log=print, mask_fn=None):
    """Find colored cubes before containers appear: use last frame before `upto` with low points."""
    best = None
    for i in range(upto - 1, -1, -1):
        d = data[i]
        if len(d["low"]) > 50:
            best = i
            break
    if best is None:
        return [], None
    d = dict(data[best])
    if mask_fn is not None and len(d["low"]):
        keep = mask_fn(d["low"].astype(float))
        d["low"], d["lowc"] = d["low"][keep], d["lowc"][keep]
    bl = blobs(d["low"], min_size=8)
    cubes = []
    for b in bl:
        rr, cc = b["cells"]
        r, c = to_cells(d["low"].astype(float))
        cellset = set(zip(rr.tolist(), cc.tolist()))
        sel = np.array([(a, bb) in cellset for a, bb in zip(r.tolist(), c.tolist())])
        col = np.median(d["lowc"][sel], 0) if sel.any() else np.array([0, 0, 0])
        cubes.append(dict(xy=b["xy"], size=b["size"], rgb=col.tolist(), color=classify_color(col)))
    log(f"cubes from frame {best}: " + ", ".join(f"{c['color']}@{np.round(c['xy'],3).tolist()} n={c['size']}" for c in cubes))
    return cubes, best

def find_init(data, min_size=60, stable=6, log=print):
    """First frame index after which the count of big top blobs is stable for `stable` frames."""
    counts = []
    for d in data:
        counts.append(len([b for b in blobs(d["top"]) if b["size"] >= min_size]))
    for i in range(len(counts) - stable):
        if counts[i] > 0 and all(counts[j] == counts[i] for j in range(i, i + stable)):
            # wait until top height settled
            return i + stable - 1, counts[i]
    return None, 0

def static_tall_mask(data, frac=0.5):
    """Cells occupied by tall (z 0.10-0.16) points in >= frac of frames: static occluders."""
    from scipy import ndimage
    acc = np.zeros(GRID, int)
    for d in data:
        g = np.zeros(GRID, bool)
        if len(d["tall"]):
            r, c = to_cells(d["tall"].astype(float)); g[r, c] = True
        acc += g
    m = acc >= max(1, frac * len(data))
    m = ndimage.binary_dilation(m, iterations=3)
    return m

def make_mask_fn(m):
    if m is None or not m.any():
        return None
    def fn(pts):
        r, c = to_cells(pts)
        r = np.clip(r, 0, GRID[0] - 1); c = np.clip(c, 0, GRID[1] - 1)
        return ~m[r, c]
    return fn

def track(data, start, init_xy, radius=0.04, log=print, mask_fn=None):
    tr = Tracker(init_xy, radius)
    hist = [tr.T.copy()]
    for i in range(start + 1, len(data)):
        tr.update(data[i]["top"], mask_fn)
        hist.append(tr.T.copy())
    return np.array(hist), tr.lost

def solve_video(data, log=print):
    """Return dict with cube list, per-container color and final xy."""
    m = static_tall_mask(data)
    mask_fn = make_mask_fn(m)
    log(f"static tall mask cells={int(m.sum())}")
    if mask_fn is not None:
        nd = []
        for d in data:
            d = dict(d)
            if len(d["top"]):
                d["top"] = d["top"][mask_fn(d["top"].astype(float))]
            if len(d["low"]):
                k = mask_fn(d["low"].astype(float)); d["low"], d["lowc"] = d["low"][k], d["lowc"][k]
            nd.append(d)
        data = nd
    first_top = None
    for i, d in enumerate(data):
        if len(d["top"]) > 30:
            first_top = i
            break
    if first_top is None:
        raise RuntimeError("no containers seen in video")
    cubes, cf = detect_cubes(data, first_top, log=log)
    if not cubes:
        raise RuntimeError("no cubes seen before containers")
    init = np.array([c["xy"] for c in cubes])
    H, lost = track(data, first_top - 1, init, log=log)
    steps = np.linalg.norm(np.diff(H, axis=0), axis=2)
    log(f"first_top={first_top} n={len(cubes)} max_step={steps.max():.3f} lost_final={lost.tolist()}")
    return dict(cubes=cubes, H=H, final=H[-1], colors=[c["color"] for c in cubes], first_top=first_top,
                mask=m, lost=lost)

class Tracker:
    def __init__(self, init_xy, radius=0.04):
        self.T = np.array(init_xy, float)
        self.vel = np.zeros_like(self.T)
        self.lost = np.zeros(len(self.T), int)
        self.radius = radius

    def update(self, pts, mask_fn=None):
        T, vel, radius = self.T, self.vel, self.radius
        pts = np.asarray(pts, float)
        if mask_fn is not None and len(pts):
            pts = pts[mask_fn(pts)]
        pred = T + vel
        newT = pred.copy()
        cnt = np.zeros(len(T), int)
        if len(pts):
            for _ in range(3):
                D = np.linalg.norm(pts[:, None, :] - newT[None, :, :], axis=2)
                a = D.argmin(1)
                dm = D[np.arange(len(pts)), a]
                for k in range(len(T)):
                    s = (a == k) & (dm < radius)
                    if s.sum() >= 12:
                        newT[k] = pts[s].mean(0)
            D = np.linalg.norm(pts[:, None, :] - newT[None, :, :], axis=2)
            a = D.argmin(1); dm = D[np.arange(len(pts)), a]
            cnt = np.array([((a == k) & (dm < radius)).sum() for k in range(len(T))])
        moved = np.zeros(len(T))
        for k in range(len(T)):
            if cnt[k] >= 12:
                moved[k] = np.linalg.norm(newT[k] - T[k])
                vel[k] = 0.5 * vel[k] + 0.5 * (newT[k] - T[k])
                T[k] = newT[k]
                self.lost[k] = 0
            else:
                self.lost[k] += 1
                vel[k] = 0
        # re-acquire lost tracks from unclaimed blobs
        if len(pts) and (self.lost > 0).any():
            D = np.linalg.norm(pts[:, None, :] - T[None, :, :], axis=2)
            un = pts[D.min(1) > 0.035]
            if len(un) >= 40:
                for b in sorted(blobs(un, min_size=40), key=lambda b: -b["size"]):
                    lk = [k for k in range(len(T)) if self.lost[k] > 0]
                    if not lk:
                        break
                    dd = [np.linalg.norm(T[k] - b["xy"]) for k in lk]
                    j = int(np.argmin(dd))
                    if dd[j] < 0.25:
                        k = lk[j]
                        T[k] = b["xy"]; vel[k] = 0; self.lost[k] = 0
        return moved, cnt

def points_from_obs(meta, imgs, use_wrist=None):
    from geom import backproject
    use_wrist = USE_WRIST if use_wrist is None else use_wrist
    d = imgs[("front", "depth")].astype(np.float64) * meta["depth"]["front"].get("meters_per_unit", 0.001)
    P = backproject(d, meta["camera_intrinsics"]["front"], meta["camera_poses"]["front"]).reshape(-1, 3)
    if use_wrist and ("wrist", "depth") in imgs:
        wd = imgs[("wrist", "depth")].astype(np.float64) * meta["depth"]["wrist"].get("meters_per_unit", 0.001)
        WP = backproject(wd, meta["camera_intrinsics"]["wrist"], meta["camera_poses"]["wrist"])[::2, ::2].reshape(-1, 3)
        P = np.concatenate([P, WP])
    ok = in_ws(P) & np.isfinite(P[:, 2])
    z = P[:, 2]
    return P[ok & (z > 0.055) & (z < 0.095), :2]
