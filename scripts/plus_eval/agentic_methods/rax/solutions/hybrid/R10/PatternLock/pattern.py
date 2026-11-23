import json
import numpy as np
from geom import WS, world_points, load_frame
from grid import detect_discs, refine_grid, nearest_index


def frame_points(fr, camera="front"):
    rgb, dep = load_frame(fr, camera)
    P = world_points(dep, fr["camera_intrinsics"][camera]["matrix"], fr["camera_poses"][camera])
    return rgb, dep, P


def grid_from_frame(fr, log=print):
    rgb, dep, P = frame_points(fr)
    ds = detect_discs(P, dep > 0)
    cnts = np.array([n for c, n, uv in ds]) if ds else np.array([])
    cs = [c for c, n, uv in ds]
    if len(cs) > 25:
        med = np.median(cnts)
        keep = [i for i in range(len(ds)) if cnts[i] < 2.5 * med]
        cs = [cs[i] for i in keep]
    xy = np.array([c[:2] for c in cs])
    if len(xy) > 25:
        # drop points far from the median location
        mid = np.median(xy, 0)
        d = np.linalg.norm(xy - mid, axis=1)
        xy = xy[np.argsort(d)[:25]]
    g = refine_grid(xy)
    g["top_z"] = float(np.median([c[2] for c in cs])) if cs else 0.016
    log("grid: n_det=%d matched=%d spacing=%.4f theta=%.2fdeg center=%s" % (
        len(xy), len(g["matched"]), g["spacing"], np.degrees(g["theta"]), np.round(g["center"], 4).tolist()))
    return g


def red_mask(rgb):
    r, gch, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 120) & (r > 2.0 * gch) & (r > 2.0 * b)


def extract_pattern(manifest_path, log=print, min_px=3, gap_frames=12):
    m = json.loads((WS / manifest_path).read_text())
    frames = [f for f in m["frames"] if f["phase"] == "video_demo"]
    frames.sort(key=lambda f: f["frame_index"])
    if not frames:
        return None, [], None
    g = grid_from_frame(frames[0], log)
    s = g["spacing"]
    keys = list(g["lattice"])
    L = np.array([g["lattice"][k] for k in keys])
    seq = []
    last_red = {}  # key -> last frame index seen red
    per_frame = []
    for fi, fr in enumerate(frames):
        try:
            rgb, dep, P = frame_points(fr)
        except Exception as e:  # missing file
            log("frame %s load error %s" % (fr.get("frame_index"), e))
            continue
        mk = red_mask(rgb) & (dep > 0)
        reds = []
        if mk.sum() >= min_px:
            pts = P[mk]
            pts = pts[(pts[:, 2] > -0.01) & (pts[:, 2] < 0.06)]
            if len(pts):
                d = np.linalg.norm(pts[:, None, :2] - L[None], axis=-1)
                idx = d.argmin(1)
                ok = d[np.arange(len(pts)), idx] < 0.45 * s
                cnt = np.bincount(idx[ok], minlength=len(keys))
                reds = [keys[i] for i in np.nonzero(cnt >= min_px)[0]]
        per_frame.append((fr["frame_index"], reds))
        # new activations: red now and not red within the last gap_frames frames
        new = [k for k in reds if not (k in last_red and fi - last_red[k] <= gap_frames)]
        # order multiple simultaneous new ones (rare): keep detection order
        for k in new:
            if not seq or seq[-1][0] != k:
                seq.append((k, fr["frame_index"]))
        for k in reds:
            last_red[k] = fi
    log("pattern: %s" % [(k, f) for k, f in seq])
    return g, seq, per_frame
