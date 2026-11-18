"""Observation images, depth->world mapping and tabletop scene analysis."""
import math
import numpy as np

from io_utils import (log, reply_json, reply_images, find_key, load_manifest, frame_files,
                      load_rgb, load_depth, depth_from_image, files_from_json, find_state)
from geom import DepthMap, fit_depth_map, calib_depth_map, quat_wxyz_to_mat, min_area_yaw

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

CAMS = ("front", "wrist")


# ---------------------------------------------------------------- images of an observation
class Obs:
    def __init__(self):
        self.json = {}
        self.step_id = None
        self.state = None
        self.rgb = {}
        self.depth = {}
        self.src = {}

    def ok(self, cam):
        return cam in self.depth and self.depth[cam] is not None


def obs_from_reply(result, prefer_manifest=True):
    """Build an Obs from a robomme_observe / robomme_step reply."""
    o = Obs()
    j = reply_json(result)
    o.json = j
    o.state = find_state(j)
    sid = find_key(j, "step_id")
    o.step_id = sid if isinstance(sid, int) else None
    # 1) explicit files of the current observation in the JSON
    try:
        files = files_from_json(j)
        _load_files(o, files, "json-files")
    except Exception as e:
        log("files_from_json failed", err=str(e))
    # 2) manifest: last frame matching this step id
    if prefer_manifest and not (o.ok("front") and "front" in o.rgb):
        try:
            man, mpath = load_manifest(j)
            if man and man.get("frames"):
                frames = man["frames"]
                cand = [f for f in frames if f.get("phase", "execution") == "execution"] or frames
                pick = None
                if o.step_id is not None:
                    same = [f for f in cand if f.get("step_id") == o.step_id]
                    if same:
                        pick = max(same, key=lambda f: f.get("frame_index", 0))
                if pick is None and o.step_id is None:
                    pick = max(cand, key=lambda f: f.get("frame_index", 0))
                if pick is not None:
                    _load_files(o, frame_files(pick, mpath), "manifest")
        except Exception as e:
            log("manifest frame load failed", err=str(e))
    # 3) inline image blocks
    if not (o.ok("front") and "front" in o.rgb):
        try:
            _load_inline(o, result, j)
        except Exception as e:
            log("inline decode failed", err=str(e))
    return o


def _load_files(o, files, tag):
    for (cam, kind), (path, mpu) in files.items():
        if cam not in CAMS:
            continue
        try:
            if kind == "rgb" and cam not in o.rgb:
                o.rgb[cam] = load_rgb(path)
                o.src[cam + "_rgb"] = tag
            elif kind == "depth" and cam not in o.depth:
                d = load_depth(path, mpu)
                if d is not None:
                    o.depth[cam] = d
                    o.src[cam + "_depth"] = tag
        except Exception as e:
            log("file load failed", path=str(path), err=str(e))


def _cam_of(meta, text_hint):
    s = (str(meta) + " " + (text_hint or "")).lower()
    for c in CAMS:
        if c in s:
            return c
    return None


def _load_inline(o, result, j):
    imgs = reply_images(result)
    if not imgs:
        return
    # descriptor list in the JSON?
    desc = None
    for key in ("attachments", "images", "image_attachments", "media"):
        d = find_key(j, key)
        if isinstance(d, list) and len(d) == len(imgs) and all(isinstance(x, dict) for x in d):
            desc = d
            break
    blocks = result.get("content", [])
    rgb_order, depth_order = [], []
    for k, im in enumerate(imgs):
        cam = None
        kind = None
        if desc is not None:
            cam = _cam_of({kk: vv for kk, vv in desc[k].items() if kk in ("camera", "name", "view")}, "")
            kind = "depth" if "depth" in str(desc[k].get("kind", desc[k].get("type", ""))).lower() else None
        if cam is None:
            prev = blocks[im["index"] - 1] if im["index"] > 0 else {}
            hint = prev.get("text", "") if isinstance(prev, dict) and prev.get("type") == "text" and len(prev.get("text", "")) < 300 else ""
            cam = _cam_of(im["meta"], hint)
        img = im["img"]
        if img.mode in ("I;16", "I;16B", "I;16L", "I", "F") or kind == "depth":
            d = depth_from_image(img, 0.001)
            if d is not None:
                depth_order.append((cam, d))
        else:
            rgb_order.append((cam, np.asarray(img.convert("RGB"))))
    for lst, store, tag in ((rgb_order, o.rgb, "_rgb"), (depth_order, o.depth, "_depth")):
        unnamed = [CAMS[0], CAMS[1]]
        for cam, arr in lst:
            if cam is not None and cam in unnamed:
                unnamed.remove(cam)
        for cam, arr in lst:
            c = cam
            if c is None:
                if not unnamed:
                    continue
                c = unnamed.pop(0)
            if c not in store:
                store[c] = arr
                o.src[c + tag] = "inline"


# ---------------------------------------------------------------- calibration
def _mat3(x):
    try:
        a = np.asarray(x, dtype=float)
        if a.shape == (3, 3):
            return a
        if a.size == 9:
            return a.reshape(3, 3)
        if a.shape == (3, 4) or a.shape == (4, 4):
            return a[:3, :3]
    except Exception:
        pass
    return None


def parse_intrinsics(j, cam):
    ci = find_key(j, "camera_intrinsics")
    if ci is None:
        return None
    e = ci.get(cam) if isinstance(ci, dict) else None
    if e is None and isinstance(ci, dict):
        for k, v in ci.items():
            if cam in str(k).lower():
                e = v
    if e is None:
        return None
    if isinstance(e, dict):
        for k in ("K", "matrix", "intrinsic", "intrinsics", "intrinsic_cv", "camera_matrix"):
            if k in e:
                m = _mat3(e[k])
                if m is not None:
                    return m
        if all(k in e for k in ("fx", "fy", "cx", "cy")):
            return np.array([[e["fx"], 0, e["cx"]], [0, e["fy"], e["cy"]], [0, 0, 1.0]], dtype=float)
        return None
    return _mat3(e)


def parse_pose(j, cam):
    cp = find_key(j, "camera_poses")
    if cp is None:
        return None
    e = cp.get(cam) if isinstance(cp, dict) else None
    if e is None and isinstance(cp, dict):
        for k, v in cp.items():
            if cam in str(k).lower():
                e = v
    if not isinstance(e, dict):
        try:
            a = np.asarray(e, dtype=float)
            if a.shape == (4, 4):
                return a[:3, :3], a[:3, 3]
        except Exception:
            pass
        return None
    pos = None
    for k in ('position_m', "position", "p", "translation", "xyz", "pos"):
        if k in e:
            pos = np.asarray(e[k], dtype=float)[:3]
    q = None
    for k in ("quaternion_wxyz", "wxyz", "quaternion", "q", "orientation"):
        if k in e:
            q = e[k]
    if pos is not None and q is not None:
        return quat_wxyz_to_mat(q), pos
    for k in ("matrix", "cam2world", "pose", "transform"):
        if k in e:
            a = np.asarray(e[k], dtype=float)
            if a.shape == (4, 4):
                return a[:3, :3], a[:3, 3]
    return None


def calib_map(j, cam):
    K = parse_intrinsics(j, cam)
    P = parse_pose(j, cam)
    if K is None or P is None:
        return None
    try:
        return calib_depth_map(K, P[0], P[1], "opengl")
    except Exception:
        return None


# ---------------------------------------------------------------- mapping fit via the official tool
def parse_p2p(result, n):
    j = reply_json(result)
    pts = []
    frontier = [j]
    found = []
    for _ in range(5):
        nxt = []
        for o in frontier:
            if isinstance(o, list) and len(o) >= 1 and all(isinstance(x, dict) for x in o):
                ok = [x for x in o if _vec3(x) is not None or "valid" in x]
                if len(ok) == len(o) and len(o) >= 1:
                    found = o
                    break
            if isinstance(o, dict):
                nxt.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                nxt.extend(v for v in o if isinstance(v, (dict, list)))
        if found:
            break
        frontier = nxt
    for x in found:
        v = _vec3(x)
        valid = x.get("valid", True)
        pts.append(np.asarray(v, dtype=float) if (v is not None and valid not in (False, "false")) else None)
    if len(pts) != n:
        log("p2p count mismatch", got=len(pts), want=n)
    return pts, j


def _vec3(x):
    for k in ("position", "world_position", "world", "point", "xyz", "position_m", "world_point"):
        v = x.get(k)
        if isinstance(v, list) and len(v) == 3 and all(isinstance(a, (int, float)) for a in v):
            return v
        if isinstance(v, dict) and all(a in v for a in ("x", "y", "z")):
            return [v["x"], v["y"], v["z"]]
    return None


def choose_fit_pixels(depth, n=30, rng=None):
    H, W = depth.shape
    rng = rng or np.random.default_rng(0)
    d = depth
    sel = []
    grid = int(math.ceil(math.sqrt(n * 1.6)))
    us = np.linspace(12, W - 13, grid)
    vs = np.linspace(12, H - 13, grid)
    cands = []
    for v in vs:
        for u in us:
            ui, vi = int(round(u)), int(round(v))
            win = d[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            if win.shape != (5, 5) or not np.isfinite(win).all():
                continue
            # locally planar: small second differences (an oblique plane has a large range but is linear)
            c = win[2, 2]
            if abs(win[2, 1] + win[2, 3] - 2 * c) > 0.0015 or abs(win[1, 2] + win[3, 2] - 2 * c) > 0.0015:
                continue
            if abs(win[2, 0] + win[2, 4] - 2 * c) > 0.003 or abs(win[0, 2] + win[4, 2] - 2 * c) > 0.003:
                continue
            cands.append((ui, vi))
    if len(cands) > n:
        idx = rng.choice(len(cands), n, replace=False)
        cands = [cands[i] for i in sorted(idx)]
    return cands


def fit_camera_map(ctx, obs, cam, n=30):
    """Fit world = M [u d, v d, d] + t using robomme_pixel_to_position on the current observation."""
    d = obs.depth.get(cam)
    if d is None or obs.step_id is None:
        return None
    pix = choose_fit_pixels(d, n)
    if len(pix) < 8:
        log("fit: too few smooth pixels", cam=cam, n=len(pix))
        return None
    sel = [{"camera": cam, "u": float(u), "v": float(v)} for (u, v) in pix]
    res = ctx.call("robomme_pixel_to_position", observation_step_id=int(obs.step_id), selections=sel,
                   depth_window_radius=0)
    pts, j = parse_p2p(res, len(sel))
    uvd, W = [], []
    for (u, v), p in zip(pix, pts):
        if p is None:
            continue
        uvd.append((u, v, d[v, u]))
        W.append(p)
    if len(uvd) < 8:
        log("fit: too few valid tool points", cam=cam, n=len(uvd), sample=str(j)[:400])
        return None
    out = fit_depth_map(uvd, W)
    if out is None:
        return None
    m, keep = out
    log("fit camera map", cam=cam, n=len(uvd), inliers=int(keep.sum()), median_res=m.residual)
    if m.residual is None or m.residual > 0.01 or keep.sum() < 8:
        return None
    return m


# ---------------------------------------------------------------- components
def label(mask):
    if _ndi is not None:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        return lab, n
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=int)
    n = 0
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lab[y, x] == 0:
                n += 1
                stack = [(y, x)]
                lab[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n
                                stack.append((yy, xx))
    return lab, n


# ---------------------------------------------------------------- scene analysis
def table_height(P, valid):
    z = P[..., 2][valid]
    if z.size < 100:
        return None
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    if hi - lo < 1e-4:
        return float(np.median(z))
    bins = np.arange(lo, hi + 0.002, 0.002)
    hist, edges = np.histogram(z, bins=bins)
    k = int(np.argmax(hist))
    m = 0.5 * (edges[k] + edges[k + 1])
    near = z[np.abs(z - m) < 0.004]
    return float(np.median(near)) if near.size else float(m)


def analyze_scene(P, rgb=None, tz=None, ee=None):
    """Segment the tabletop into raised components and flat coloured blobs."""
    H, W, _ = P.shape
    valid = np.isfinite(P).all(-1)
    if tz is None:
        tz = table_height(P, valid)
    if tz is None:
        return None
    z = P[..., 2]
    h = np.where(valid, z - tz, np.nan)
    table = valid & (np.abs(h) < 0.005)
    if table.sum() < 50:
        return None
    txy = P[table][:, :2]
    xlo, ylo = np.percentile(txy, 0.5, axis=0) - 0.02
    xhi, yhi = np.percentile(txy, 99.5, axis=0) + 0.02
    inside = valid & (P[..., 0] > xlo) & (P[..., 0] < xhi) & (P[..., 1] > ylo) & (P[..., 1] < yhi)
    hh = np.nan_to_num(h, nan=-1.0)
    raised = inside & (hh > 0.010)
    lab, n = label(raised)
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        cnt = int(m.sum())
        if cnt < 6:
            continue
        pts = P[m]
        hs = h[m]
        top = float(np.percentile(hs, 90))
        hmax = float(np.percentile(hs, 98))
        topm = hs > top - 0.008
        tpts = pts[topm]
        cxy = tpts[:, :2].mean(0) if len(tpts) else pts[:, :2].mean(0)
        yaw, ext = min_area_yaw(tpts[:, :2]) if len(tpts) >= 5 else (None, None)
        allext = (np.percentile(pts[:, :2], 97, axis=0) - np.percentile(pts[:, :2], 3, axis=0)).tolist()
        vs, us = np.nonzero(m)
        col = rgb[m].mean(0).tolist() if rgb is not None else None
        comps.append({"n": cnt, "xy": cxy, "top": top, "hmax": hmax, "yaw": yaw, "ext": ext, "allext": allext,
                      "uv": [float(us.mean()), float(vs.mean())], "color": col,
                      "touches_border": bool(us.min() <= 0 or vs.min() <= 0 or us.max() >= W - 1 or vs.max() >= H - 1)})
    blobs = []
    tcol = None
    if rgb is not None:
        try:
            tcol = np.median(rgb[table].astype(float), axis=0)
            flat = inside & (np.abs(np.nan_to_num(h, nan=1.0)) < 0.008)
            cd = np.linalg.norm(rgb.astype(float) - tcol, axis=-1)
            fm = flat & (cd > 45)
            labf, nf = label(fm)
            for i in range(1, nf + 1):
                m = labf == i
                cnt = int(m.sum())
                if cnt < 12:
                    continue
                pts = P[m]
                vs, us = np.nonzero(m)
                blobs.append({"n": cnt, "xy": pts[:, :2].mean(0), "color": rgb[m].astype(float).mean(0).tolist(),
                              "ext": (pts[:, :2].max(0) - pts[:, :2].min(0)).tolist(),
                              "uv": [float(us.mean()), float(vs.mean())]})
        except Exception as e:
            log("blob analysis failed", err=str(e))
    return {"tz": tz, "comps": comps, "blobs": blobs, "table_color": tcol, "h": h,
            "table_frac": float(table.mean())}


def pick_cube(scene, prior_xy=None, exp_top=None, avoid_xy=(), ee=None):
    """Choose the cube among raised components."""
    if scene is None:
        return None
    best = None
    for c in scene["comps"]:
        top = c["top"]
        if not (0.015 <= top <= 0.09) or c["hmax"] > 0.11:
            continue
        ext = c["ext"] or c["allext"]
        if ext is None or max(ext) > 0.085 or max(c["allext"]) > 0.12:
            continue
        if c["n"] < 8:
            continue
        s = 0.0
        if exp_top is not None:
            s += abs(top - exp_top) * 20.0
        else:
            s += abs(top - 0.04) * 10.0
        if prior_xy is not None:
            s += float(np.linalg.norm(np.asarray(c["xy"]) - prior_xy)) * 10.0
        for a in avoid_xy:
            if a is not None and np.linalg.norm(np.asarray(c["xy"]) - np.asarray(a)) < 0.04:
                s += 3.0
        if ee is not None and np.linalg.norm(np.asarray(c["xy"]) - np.asarray(ee[:2])) < 0.03 and top > 0.06:
            s += 2.0
        if best is None or s < best[0]:
            best = (s, c)
    return None if best is None else dict(best[1], score=best[0])
