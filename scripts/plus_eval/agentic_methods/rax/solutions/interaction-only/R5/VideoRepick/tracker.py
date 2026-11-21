"""Track the picked object through the conditioning video using the static front camera."""
import json
import numpy as np
from vision import green_mask, backproject, load_rgb, load_depth

N_OBJ_DEFAULT = 6


def frame_files(f):
    return {(x["camera"], x["kind"]): x["path"] for x in f["files"]}


PIXEL_EXCLUDE = None  # static front-camera pixel mask (button), built from video frame 0


def build_button_pixel_mask(f, root, center, radius=0.05, dilate=3):
    """Front-camera pixels of the button (low saturation, raised 8-45 mm, near centre), dilated."""
    from vision import rgb_to_hsv, backproject
    files = frame_files(f)
    rgb = load_rgb(f"{root}/{files[('front', 'rgb')]}")
    dep = load_depth(f"{root}/{files[('front', 'depth')]}")
    H, W = dep.shape
    vs, us = np.nonzero(dep > 0)
    pw = backproject(us.astype(float), vs.astype(float), dep[vs, us],
                     f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    hsv = rgb_to_hsv(rgb[vs, us].astype(float) / 255.0)
    sel = (pw[:, 2] > 0.006) & (pw[:, 2] < 0.05) & (hsv[:, 1] < 0.2) & \
        (np.linalg.norm(pw[:, :2] - np.asarray(center), axis=1) < radius)
    m = np.zeros((H, W), bool)
    m[vs[sel], us[sel]] = True
    out = m.copy()
    for dy in range(-dilate, dilate + 1):
        for dx in range(-dilate, dilate + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def front_points(f, root):
    files = frame_files(f)
    rgb = load_rgb(f"{root}/{files[('front', 'rgb')]}")
    dep = load_depth(f"{root}/{files[('front', 'depth')]}")
    m = green_mask(rgb) & (dep > 0)
    if PIXEL_EXCLUDE is not None and PIXEL_EXCLUDE.shape == m.shape:
        m &= ~PIXEL_EXCLUDE
    vs, us = np.nonzero(m)
    if len(vs) == 0:
        return np.zeros((0, 3))
    pw = backproject(us.astype(float), vs.astype(float), dep[vs, us],
                     f["camera_intrinsics"]["front"]["matrix"], f["camera_poses"]["front"])
    return pw


EXCLUDE = []  # list of (xy, radius) regions (button)


WRIST_EXCLUDE = []  # world (xy, r) regions removed from wrist points (button)
USE_WRIST = True


def wrist_points(f, root):
    files = frame_files(f)
    if ("wrist", "rgb") not in files or ("wrist", "depth") not in files:
        return np.zeros((0, 3))
    rgb = load_rgb(f"{root}/{files[('wrist', 'rgb')]}")
    dep = load_depth(f"{root}/{files[('wrist', 'depth')]}")
    m = green_mask(rgb) & (dep > 0)
    vs, us = np.nonzero(m)
    if len(vs) == 0:
        return np.zeros((0, 3))
    pw = backproject(us.astype(float), vs.astype(float), dep[vs, us],
                     f["camera_intrinsics"]["wrist"]["matrix"], f["camera_poses"]["wrist"])
    for c, r in WRIST_EXCLUDE:
        pw = pw[np.linalg.norm(pw[:, :2] - np.asarray(c), axis=1) > r]
    return pw


def fused_xy(f, root, use_wrist=None):
    """Table-object xy points from front (+ wrist when the arm is high and still-ish) cameras."""
    xy = table_points(front_points(f, root))
    uw = USE_WRIST if use_wrist is None else use_wrist
    if uw and f["state"][2] > 0.15:
        w = table_points(wrist_points(f, root))
        if len(w):
            # thin wrist points to a 3 mm grid so close-up views do not dominate
            key = np.floor(w / 0.003).astype(np.int64)
            _, idx = np.unique(key[:, 0] * 100003 + key[:, 1], return_index=True)
            w = w[idx]
            xy = np.concatenate([xy, w]) if len(xy) else w
    return xy


def grid_area(xy, cell=0.004):
    if len(xy) == 0:
        return 0
    key = np.floor(xy / cell).astype(np.int64)
    return len(np.unique(key[:, 0] * 100003 + key[:, 1]))


def init_centers_area(xy):
    comps = [c for c in components_xy(xy) if len(c) >= 10]
    cents = [xy[c].mean(0) for c in comps]
    areas = [grid_area(xy[c]) for c in comps]
    return cents, areas, comps


def table_points(pw, zmax=0.07):
    sel = (pw[:, 2] < zmax) & (pw[:, 2] > 0.003) & (pw[:, 0] > -0.45) & (pw[:, 0] < 0.3) & (np.abs(pw[:, 1]) < 0.4)
    for c, r in EXCLUDE:
        sel &= np.linalg.norm(pw[:, :2] - np.asarray(c), axis=1) > r
    return pw[sel][:, :2]


def components_xy(xy, link=0.012):
    """Simple single-link clustering of xy points on a grid; returns list of index arrays."""
    if len(xy) == 0:
        return []
    cell = np.floor(xy / link).astype(int)
    keys = {}
    for i, k in enumerate(map(tuple, cell)):
        keys.setdefault(k, []).append(i)
    seen = {}
    comps = []
    for k in keys:
        if k in seen:
            continue
        stack = [k]; seen[k] = len(comps); members = []
        while stack:
            c = stack.pop(); members.extend(keys[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in keys and nb not in seen:
                        seen[nb] = len(comps); stack.append(nb)
        comps.append(np.array(members))
    return comps


def init_centers(xy, n):
    comps = components_xy(xy)
    comps = [c for c in comps if len(c) >= 15]
    cents = [xy[c].mean(0) for c in comps]
    sizes = [len(c) for c in comps]
    return np.array(cents), sizes


def kmeans_track(xy, centers, iters=8):
    c = centers.copy()
    for _ in range(iters):
        d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for k in range(len(c)):
            sel = a == k
            if sel.sum() >= 5:
                c[k] = xy[sel].mean(0)
    d = ((xy[:, None, :] - c[None]) ** 2).sum(-1)
    a = d.argmin(1)
    counts = np.bincount(a, minlength=len(c))
    return c, counts


def find_release_frame(frames):
    """Index into frames (video_demo only) where the pick-and-place ends: gripper reopened after a close,
    and arm raised. Returns (index, release_xy)."""
    closed_seen = False
    release_xy = None
    for i, f in enumerate(frames):
        s = f["state"]
        g = s[6] + s[7]
        if g < 0.05 and not closed_seen:
            closed_seen = True
        if closed_seen and g > 0.075 and release_xy is None:
            release_xy = np.array(s[:2])
        if release_xy is not None and s[2] > 0.12:
            return i, release_xy
    return None, release_xy


def settle_frame(video, ri):
    """First frame after release where the arm has been still (and high) for 3 frames."""
    for i in range(ri + 1, len(video) - 6):
        p = [np.array(video[i + k]["state"][:3]) for k in range(6)]
        if p[0][2] > 0.15 and np.linalg.norm(p[5] - p[0]) < 0.002:
            return i
    return min(ri + 2, len(video) - 1)


def track(manifest_path, root="/workspace", log=print):
    m = json.load(open(manifest_path))
    frames = sorted(m["frames"], key=lambda f: f["frame_index"])
    video = [f for f in frames if f["phase"] == "video_demo"]
    execf = [f for f in frames if f["phase"] != "video_demo"]
    ri, release_xy = find_release_frame(video)
    if ri is None:
        log(f"no release found; release_xy={release_xy}")
        ri = 0
    start = settle_frame(video, ri)
    # frame 0: objects untouched; count by occupied area
    xy0 = fused_xy(video[0], root)
    c0, a0, comps0 = init_centers_area(xy0)
    typical = float(np.median(a0)) if len(a0) else 1.0
    ks = [max(1, int(round(a / typical))) for a in a0]
    seeds = []
    for c, k, cc in zip(c0, ks, comps0):
        if k == 1:
            seeds.append(np.asarray(c))
        else:
            pts = xy0[cc]
            for j in range(k):
                seeds.append(pts[int(j * len(pts) / k)])
    c0, _ = kmeans_track(xy0, np.array(seeds, float))
    n0 = len(c0)
    xy = fused_xy(video[start], root)
    cs, as_, _ = init_centers_area(xy)
    log(f"release frame {ri} xy={release_xy}, start {start}, n0 {n0} areas0 {a0}, start comps {len(cs)} areas {as_}")
    if len(cs) == n0:
        centers = np.array(cs, float)
    else:
        centers, _ = kmeans_track(xy, np.array(c0, float))
    n = len(centers)
    if release_xy is None:
        release_xy = centers.mean(0)
    dists = np.linalg.norm(centers - release_xy, axis=1)
    target = int(np.argmin(dists))
    srt = np.sort(dists)
    # displacement of each start centre from frame-0 centres (the re-placed object moved)
    disp = np.array([np.min(np.linalg.norm(np.asarray(c0) - c, axis=1)) for c in centers])
    moved = int(np.argmax(disp))
    log(f"target init {centers[target]} dist {srt[0]:.3f} second {srt[1] if len(srt) > 1 else -1:.3f}; "
        f"moved obj {moved} disp {disp[moved]:.3f} (target disp {disp[target]:.3f})")
    if srt[0] > 0.05 and disp[moved] > 0.01:
        log("release match weak; using the moved object")
        target = moved
    hist = [centers.copy()]
    prev = centers.copy()
    for f in video[start + 1:] + execf[:1]:
        xy = fused_xy(f, root)
        if len(xy) < 20:
            hist.append(centers.copy()); continue
        pred = centers + 0.7 * (centers - prev)
        new, counts = kmeans_track(xy, pred)
        prev = centers; centers = new
        hist.append(centers.copy())
    return dict(target=target, centers=centers, target_xy=centers[target], hist=np.array(hist),
                release_xy=release_xy, start=start, n=n)
