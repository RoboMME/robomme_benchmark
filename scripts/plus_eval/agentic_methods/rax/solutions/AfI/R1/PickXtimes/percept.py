"""RGB-D perception: camera parsing, back-projection validated against the
robomme_pixel_to_position tool, table-plane estimation, object segmentation.

All tunables are named constants; none was tuned on a live scene (no live scene was
available before the single rehearsal), they are chosen from geometry: 256x256 images,
a ~4 cm cube, a flat target disk, a button standing on the table.
"""
import math

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from geom import quat_wxyz_to_mat
from mcpio import find_all, is_numlist, load_image

ABOVE_TABLE_MIN = 0.007      # m, points higher than this above the table are objects
ABOVE_TABLE_MAX = 0.16       # m, higher points belong to the arm
FLAT_BAND = 0.0065           # m, |z - table| below this counts as table/target level
VOXEL = 0.004                # m, downsampling voxel for clustering
CLUSTER_R = 0.0085           # m, neighbour radius for connected components
MIN_CLUSTER_PTS = 6          # voxels
MAX_OBJ_EXTENT = 0.16        # m, larger components are arm / clutter
CALIB_ACCEPT_M = 0.02        # m, max median disagreement with the tool
SPLIT_MIN_EXTENT = 0.065    # m, wider low components are checked for objects on top
TARGET_BAND_MAX = 0.017      # m, highest target-disk surface considered


# --------------------------------------------------------------------------
# parsing camera metadata
# --------------------------------------------------------------------------
def per_camera(v):
    """Map camera name -> value from dict-or-list layouts."""
    out = {}
    if isinstance(v, dict):
        for k, x in v.items():
            kl = str(k).lower()
            for cam in ("front", "wrist"):
                if cam in kl:
                    out[cam] = x
        if not out and ("camera" in v or "name" in v):
            name = str(v.get("camera") or v.get("name")).lower()
            for cam in ("front", "wrist"):
                if cam in name:
                    out[cam] = v
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                name = str(x.get("camera") or x.get("name") or "").lower()
                for cam in ("front", "wrist"):
                    if cam in name:
                        out[cam] = x
    return out


def parse_K(v):
    if v is None:
        return None
    if isinstance(v, dict):
        if all(k in v for k in ("fx", "fy", "cx", "cy")):
            return np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1.0]], float)
        for k in ("K", "matrix", "intrinsic_cv", "intrinsics", "camera_matrix", "k", "intrinsic"):
            if k in v:
                r = parse_K(v[k])
                if r is not None:
                    return r
        for x in v.values():
            if isinstance(x, (list, dict)):
                r = parse_K(x)
                if r is not None:
                    return r
        return None
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return None
    if a.shape == (3, 3):
        return a
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:
        fx, fy, cx, cy = a.ravel()
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return None


def parse_pose(v):
    """Return (R, t, convention string) of a camera pose."""
    if v is None:
        return None
    conv = ""
    if isinstance(v, dict):
        for k in ("convention", "frame_convention", "camera_convention", "axes"):
            if isinstance(v.get(k), str):
                conv = v[k].lower()
        pos = None
        for k in ('position_m', "position", "p", "translation", "t", "xyz", "pos"):
            if is_numlist(v.get(k), 3):
                pos = np.array(v[k], float)
        q = None
        for k in ("quaternion_wxyz", "wxyz", "q", "quat", "quaternion", "orientation"):
            if is_numlist(v.get(k), 4):
                q = v[k]
        if pos is not None and q is not None:
            return quat_wxyz_to_mat(q), pos, conv
        for k in ("matrix", "pose", "cam2world", "camera_to_world", "extrinsic", "T"):
            if k in v:
                r = parse_pose(v[k])
                if r is not None:
                    return r[0], r[1], conv or r[2]
        for x in v.values():
            if isinstance(x, (dict, list)):
                r = parse_pose(x)
                if r is not None:
                    return r[0], r[1], conv or r[2]
        return None
    try:
        a = np.array(v, dtype=float)
    except Exception:
        return None
    if a.shape == (4, 4) or a.size == 16:
        a = a.reshape(4, 4)
        return a[:3, :3], a[:3, 3], conv
    if a.shape == (3, 4):
        return a[:, :3], a[:, 3], conv
    if a.size == 7:
        return quat_wxyz_to_mat(a[3:]), a[:3], conv
    return None


def _img_kind(im):
    if im is None:
        return None
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return "depth"
    if im.mode in ("RGB", "RGBA", "P", "L"):
        return "rgb"
    return None


def _cam_from_text(*texts):
    """Camera named by a label, only when exactly one camera is mentioned."""
    for t in texts:
        tl = str(t).lower()
        w = ("wrist" in tl) or ("hand" in tl)
        f = ("front" in tl) or ("third" in tl)
        if w and not f:
            return "wrist"
        if f and not w:
            return "front"
    return None


def _attachment_list(meta, n):
    """A metadata list describing the image attachments in order, if any."""
    best = None
    for d in _walk_dicts(meta):
        for k, v in d.items():
            if isinstance(v, list) and len(v) == n and n > 0 and all(isinstance(x, dict) for x in v):
                if all(any(kk in x for kk in ("camera", "name", "view")) for x in v):
                    best = v
    return best


def extract_frames(reply, log=None):
    """Return {camera: {'rgb': HxWx3 uint8, 'depth': HxW float m}} for the reply's
    current observation, from image blocks, metadata file paths or the manifest."""
    cams = {"front": {}, "wrist": {}}
    # 1) inline image blocks
    order = {"rgb": [], "depth": []}
    att = _attachment_list(reply.meta, len(reply.images))
    for bi, blk in enumerate(reply.images):
        im = blk["img"]
        kind = _img_kind(im)
        if kind is None:
            continue
        cam = None
        if att is not None:
            a = att[bi]
            cam = _cam_from_text(a.get("camera", ""), a.get("name", ""), a.get("view", ""))
        if cam is None:
            cam = _cam_from_text(blk["info"], blk["pre"])
        txt = (str(blk["info"]) + " " + blk["pre"]).lower()
        if "depth" in txt and kind == "rgb" and im.mode == "L":
            kind = "rgb"  # 8-bit depth previews are not metric; keep as rgb candidate only
        order[kind].append((cam, im))
    for kind, lst in order.items():
        unnamed = ["front", "wrist"]
        for cam, im in lst:
            if cam is not None and cam in unnamed:
                unnamed.remove(cam)
        for cam, im in lst:
            if cam is None:
                if not unnamed:
                    continue
                cam = unnamed.pop(0)
            if kind in cams[cam]:
                continue
            cams[cam][kind] = im
    # 2) metadata file references (depth attachments / frame files)
    refs = []
    for d in _walk_dicts(reply.meta):
        p = d.get("path") or d.get("file")
        if isinstance(p, str) and p.lower().endswith((".png", ".jpg", ".jpeg")):
            refs.append(d)
    if refs:
        # prefer files of the last execution frame when they come from a manifest
        for d in refs:
            cam = _cam_from_text(d.get("camera", ""), d.get("path", ""))
            if cam is None:
                continue
            kind = str(d.get("kind", "")).lower() or ("depth" if "depth" in d.get("path", "").lower() else "rgb")
            if kind not in ("rgb", "depth"):
                continue
            # later refs override earlier ones (last frame is the current one)
            try:
                cams[cam]["_ref_" + kind] = (d.get("path") or d.get("file"), d.get("meters_per_unit"))
            except Exception:
                pass
    # 3) the frame_sequence manifest (last frame)
    ms = reply.first("manifest_path", pred=lambda v: isinstance(v, str))
    if ms:
        try:
            import json
            import os
            mp = ms if os.path.isabs(ms) else os.path.join("/workspace", ms)
            man = json.load(open(mp))
            frames = man.get("frames") or []
            if frames:
                fr = sorted(frames, key=lambda f: f.get("frame_index", 0))[-1]
                for f in fr.get("files", []):
                    cam = _cam_from_text(f.get("camera", ""))
                    kind = f.get("kind")
                    if cam and kind in ("rgb", "depth"):
                        cams[cam].setdefault("_man_" + kind, (f.get("path"), f.get("meters_per_unit")))
        except Exception as e:
            if log:
                log("manifest read failed", repr(e))
    out = {}
    for cam, d in cams.items():
        rgb = d.get("rgb")
        dep = d.get("depth")
        mpu = None
        if dep is None:
            for key in ("_ref_depth", "_man_depth"):
                if key in d:
                    try:
                        dep = load_image(d[key][0])
                        mpu = d[key][1]
                        break
                    except Exception:
                        dep = None
        if rgb is None:
            for key in ("_ref_rgb", "_man_rgb"):
                if key in d:
                    try:
                        rgb = load_image(d[key][0])
                        break
                    except Exception:
                        rgb = None
        if rgb is None or dep is None:
            continue
        rgb_a = np.array(rgb.convert("RGB"))
        dep_a = np.array(dep).astype(np.float64)
        if mpu is None:
            mpu = 0.001 if dep_a.max() > 50 else 1.0
        dep_a = dep_a * float(mpu)
        if rgb_a.shape[:2] != dep_a.shape[:2]:
            continue
        out[cam] = {"rgb": rgb_a, "depth": dep_a}
    return out


def _walk_dicts(obj, depth=0):
    if depth > 10:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:500]:
            yield from _walk_dicts(v, depth + 1)


def _search_calib(meta, words):
    """Per-camera values of the first key containing one of `words`."""
    for d in _walk_dicts(meta):
        for k, v in d.items():
            kl = str(k).lower()
            if any(w in kl for w in words) and isinstance(v, (dict, list)):
                pc = per_camera(v)
                if pc:
                    return pc
    # camera-keyed dicts holding both
    out = {}
    for d in _walk_dicts(meta):
        for k, v in d.items():
            cam = _cam_from_text(k)
            if cam and isinstance(v, dict):
                for kk, vv in v.items():
                    if any(w in str(kk).lower() for w in words):
                        out.setdefault(cam, vv)
    return out


def extract_calib(reply):
    Ks = per_camera(reply.first("camera_intrinsics") or {})
    Ps = per_camera(reply.first("camera_poses") or {})
    if len(Ks) < 2:
        Ks = {**_search_calib(reply.meta, ("intrinsic", "camera_matrix")), **Ks}
    if len(Ps) < 2:
        Ps = {**_search_calib(reply.meta, ("camera_pose", "extrinsic", "cam2world", "pose")), **Ps}
    out = {}
    for cam in ("front", "wrist"):
        K = parse_K(Ks.get(cam))
        P = parse_pose(Ps.get(cam))
        if K is not None and P is not None:
            out[cam] = {"K": K, "R": P[0], "t": P[1], "conv": P[2]}
    return out


# --------------------------------------------------------------------------
# back-projection variants
# --------------------------------------------------------------------------
VARIANTS = []
for _flip in ("none", "v", "h", "hv"):
    for _gl in (True, False):
        for _inv in (False, True):
            for _off in (0.5, 0.0):
                for _swap in (False, True):
                    VARIANTS.append({"gl": _gl, "inv": _inv, "off": _off, "swap": _swap,
                                     "flip": _flip})


def backproject(cal, depth, us, vs, var):
    """World points for pixel columns `us`, rows `vs` (arrays) under variant var."""
    K, R, t = cal["K"], cal["R"], cal["t"]
    us = np.asarray(us, float)
    vs = np.asarray(vs, float)
    if var["swap"]:
        col, row = vs, us
    else:
        col, row = us, vs
    ri = np.clip(np.round(row).astype(int), 0, depth.shape[0] - 1)
    ci = np.clip(np.round(col).astype(int), 0, depth.shape[1] - 1)
    d = depth[ri, ci]
    H, W = depth.shape
    fl = var.get("flip", "none")
    if "h" in fl:
        col = (W - 1) - col
    if "v" in fl:
        row = (H - 1) - row
    x = (col + var["off"] - K[0, 2]) / K[0, 0] * d
    y = (row + var["off"] - K[1, 2]) / K[1, 1] * d
    if var["gl"]:
        pc = np.stack([x, -y, -d], axis=-1)
    else:
        pc = np.stack([x, y, d], axis=-1)
    if var["inv"]:
        pw = (pc - t) @ R  # R^T (p - t)
    else:
        pw = pc @ R.T + t
    return pw, d


def parse_points(reply):
    """List of (xyz or None) from a pixel_to_position reply, in selection order."""
    pts = []
    lists = []
    for key in ("points", "results", "positions", "selections", "world_points", "items"):
        for _, v in find_all(reply.meta, key):
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                lists.append(v)
    if not lists:
        for _, v in find_all(reply.meta, "position"):
            pass
        # any list of dicts
        def _lists(o, d=0):
            if d > 6:
                return
            if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
                yield o
            if isinstance(o, dict):
                for x in o.values():
                    yield from _lists(x, d + 1)
            elif isinstance(o, list):
                for x in o:
                    yield from _lists(x, d + 1)
        lists = list(_lists(reply.meta))
    if not lists:
        return pts
    lst = max(lists, key=len)
    for item in lst:
        xyz = None
        valid = item.get("valid", True)
        for k in ("world_position", "position", "world", "point", "xyz", "world_point",
                  "position_m", "world_xyz", "p"):
            if is_numlist(item.get(k), 3):
                xyz = np.array(item[k], float)
                break
        if xyz is None:
            for k, v in item.items():
                if is_numlist(v, 3) and k not in ("pixel", "uv"):
                    xyz = np.array(v, float)
                    break
        if valid is False:
            xyz = None
        pts.append(xyz)
    return pts


def calibrate(ctx_call, step_id, cam, cal, frame, log, n=14):
    """Pick the back-projection variant agreeing best with the tool."""
    depth = frame["depth"]
    H, W = depth.shape
    rng = np.random.RandomState(7)
    # pick pixels with smooth valid depth, spread over the image
    cand = []
    for _ in range(400):
        u = int(rng.randint(12, W - 12))
        v = int(rng.randint(12, H - 12))
        win = depth[v - 2:v + 3, u - 2:u + 3]
        if (win <= 0).any():
            continue
        if win.max() - win.min() > 0.01:
            continue
        cand.append((u, v))
        if len(cand) >= n:
            break
    if len(cand) < 4:
        log("calib", cam, "too few smooth pixels")
        return None, None
    sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in cand]
    try:
        rep = ctx_call("robomme_pixel_to_position", observation_step_id=step_id,
                       selections=sel, depth_window_radius=0)
    except Exception as e:
        log("calib tool failed", cam, repr(e)[:300])
        return None, None
    pts = parse_points(rep)
    log("calib tool reply", cam, str(rep.meta)[:600])
    if len(pts) != len(cand):
        log("calib: point count mismatch", len(pts), len(cand))
        if not pts:
            return None, None
    best = None
    for vi, var in enumerate(VARIANTS):
        errs = []
        for (u, v), p in zip(cand, pts):
            if p is None:
                continue
            pw, _ = backproject(cal, depth, np.array([u]), np.array([v]), var)
            errs.append(np.linalg.norm(pw[0] - p))
        if len(errs) < 3:
            continue
        med = float(np.median(errs))
        if best is None or med < best[0]:
            best = (med, vi)
    if best is None:
        return None, None
    log("calib", cam, "best variant", VARIANTS[best[1]], "median err %.4f m" % best[0])
    if best[0] > CALIB_ACCEPT_M:
        return None, best[0]
    return VARIANTS[best[1]], best[0]


def flatness_variant(cal, frame, log, prefer_gl=True, return_score=False):
    """Without tool points: choose the variant whose cloud has the largest horizontal
    plane below the camera (the table)."""
    depth = frame["depth"]
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(0, W, 4), np.arange(0, H, 4))
    us = us.ravel(); vs = vs.ravel()
    ok = depth[vs, us] > 0
    us, vs = us[ok], vs[ok]
    best = None
    for var in VARIANTS:
        if var["swap"]:
            continue
        pw, d = backproject(cal, depth, us, vs, var)
        z = pw[:, 2]
        if len(z) < 50:
            continue
        zm = table_height(z)
        if zm is None:
            continue
        frac = float(np.mean(np.abs(z - zm) < 0.004))
        below = float(np.mean(z < cal["t"][2]))  # camera is above the table
        sc = frac + 0.2 * below + (0.01 if var["gl"] == prefer_gl else 0) + (0.005 if var["off"] == 0.5 else 0) \
            + (0.01 if var["flip"] == "none" and not var["inv"] else 0)
        if best is None or sc > best[0]:
            best = (sc, var, frac)
    if best is None:
        return (None, 0.0) if return_score else None
    log("flatness variant", best[1], "plane fraction %.2f" % best[2])
    if return_score:
        return best[1], best[2]
    return best[1]


def tool_cloud(ctx_call, step_id, cam, frame, stride=6):
    """Fallback: sparse world cloud from the tool on a pixel grid."""
    H, W = frame["depth"].shape
    us, vs = np.meshgrid(np.arange(stride // 2, W, stride), np.arange(stride // 2, H, stride))
    us = us.ravel()
    vs = vs.ravel()
    ok = frame["depth"][vs, us] > 0
    us, vs = us[ok], vs[ok]
    xyz = np.full((len(us), 3), np.nan)
    for i in range(0, len(us), 32):
        sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in zip(us[i:i + 32], vs[i:i + 32])]
        try:
            rep = ctx_call("robomme_pixel_to_position", observation_step_id=step_id,
                           selections=sel, depth_window_radius=0)
            pts = parse_points(rep)
        except Exception:
            pts = []
        for j, p in enumerate(pts[:len(sel)]):
            if p is not None:
                xyz[i + j] = p
    good = np.isfinite(xyz).all(axis=1)
    return xyz[good], frame["rgb"][vs[good], us[good]], us[good], vs[good]


def full_cloud(cal, frame, var, stride=1):
    depth = frame["depth"]
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    us = us.ravel()
    vs = vs.ravel()
    if var["swap"]:
        pw, d = backproject(cal, depth, vs, us, var)
    else:
        pw, d = backproject(cal, depth, us, vs, var)
    ok = (d > 0.02) & (d < 5.0) & np.isfinite(pw).all(axis=1)
    return pw[ok], frame["rgb"][vs[ok], us[ok]], us[ok], vs[ok]


# --------------------------------------------------------------------------
# scene analysis
# --------------------------------------------------------------------------
def table_height(z, ref_z=None):
    z = z[np.isfinite(z)]
    if ref_z is not None:
        z = z[z < ref_z - 0.03]
    z = z[(z > -1.5) & (z < 1.5)]
    if len(z) < 50:
        return None
    bins = np.arange(z.min(), z.max() + 0.002, 0.002)
    if len(bins) < 3:
        return float(np.median(z))
    h, e = np.histogram(z, bins=bins)
    i = int(np.argmax(h))
    mode = 0.5 * (e[i] + e[i + 1])
    near = z[np.abs(z - mode) < 0.004]
    return float(np.median(near))


def to_hsv(rgb):
    a = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)


COLOR_HUES = {  # OpenCV hue ranges (0..180)
    "red": [(0, 9), (168, 180)],
    "orange": [(9, 20)],
    "yellow": [(20, 36)],
    "green": [(36, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 128)],
    "purple": [(128, 150)],
    "pink": [(150, 168)],
}
COLOR_ALIASES = {"violet": "purple", "magenta": "pink", "grey": "gray", "teal": "cyan"}


def color_name(hsv_med):
    h, s, v = hsv_med
    if s < 70:
        if v > 170:
            return "white"
        if v < 60:
            return "black"
        return "gray"
    for name, rngs in COLOR_HUES.items():
        for lo, hi in rngs:
            if lo <= h < hi:
                return name
    return "red"


def color_score(hsv_pts, name):
    """Fraction of points whose colour matches `name`."""
    if len(hsv_pts) == 0:
        return 0.0
    h, s, v = hsv_pts[:, 0], hsv_pts[:, 1], hsv_pts[:, 2]
    name = COLOR_ALIASES.get(name, name)
    if name in COLOR_HUES:
        m = np.zeros(len(h), bool)
        for lo, hi in COLOR_HUES[name]:
            m |= (h >= lo) & (h < hi)
        m &= (s > 70) & (v > 40)
    elif name == "white":
        m = (s < 60) & (v > 170)
    elif name == "black":
        m = v < 60
    elif name == "gray":
        m = (s < 60) & (v >= 60) & (v <= 190)
    else:
        return 0.0
    return float(m.mean())


def voxel_down(xyz, rgb, vox=VOXEL):
    if len(xyz) == 0:
        return xyz, rgb
    key = np.floor(xyz / vox).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return xyz[idx], rgb[idx]


def clusters(xyz, r=CLUSTER_R, min_pts=MIN_CLUSTER_PTS):
    if len(xyz) < min_pts:
        return []
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(r, output_type="ndarray")
    n = len(xyz)
    if len(pairs) == 0:
        return []
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    out = []
    for c in range(nc):
        idx = np.nonzero(lab == c)[0]
        if len(idx) >= min_pts:
            out.append(idx)
    return out


def rect_xy(xy):
    """(center, (w, l), angle_rad) of the min-area rectangle of XY points."""
    pts = (np.asarray(xy) * 1000.0).astype(np.float32)
    if len(pts) < 3:
        c = np.asarray(xy).mean(axis=0)
        return c, (0.0, 0.0), 0.0
    (cx, cy), (w, l), ang = cv2.minAreaRect(pts)
    return np.array([cx, cy]) / 1000.0, (w / 1000.0, l / 1000.0), math.radians(ang)


def analyse(xyz, rgb, table_z, tcp=None, log=None):
    """Segment above-table objects and flat coloured regions."""
    rel = xyz[:, 2] - table_z
    keep = (rel > ABOVE_TABLE_MIN) & (rel < ABOVE_TABLE_MAX)
    if tcp is not None:
        dxy = np.linalg.norm(xyz[:, :2] - tcp[None, :2], axis=1)
        grip = (dxy < 0.075) & (xyz[:, 2] > tcp[2] - 0.02)
        keep &= ~grip
    obj_xyz, obj_rgb = voxel_down(xyz[keep], rgb[keep])
    groups = []
    for idx in clusters(obj_xyz):
        p = obj_xyz[idx]
        ext = p.max(axis=0) - p.min(axis=0)
        split = False
        if max(ext[0], ext[1]) > SPLIT_MIN_EXTENT and max(ext[0], ext[1]) < MAX_OBJ_EXTENT * 1.5:
            zr = p[:, 2] - table_z
            bins = np.arange(zr.min() - 0.001, zr.max() + 0.003, 0.002)
            if len(bins) >= 3:
                hh, ee = np.histogram(zr, bins=bins)
                zi = int(np.argmax(hh))
                zmode = 0.5 * (ee[zi] + ee[zi + 1])
                upper = zr > zmode + ABOVE_TABLE_MIN
                if zmode < 0.025 and 0.05 < upper.mean() < 0.9:
                    sub = clusters(p[upper])
                    if sub:
                        split = True
                        base_idx = idx[~upper]
                        groups.append((base_idx, table_z, "base"))
                        up_idx = idx[upper]
                        for si in sub:
                            groups.append((up_idx[si], table_z + zmode, "ontop"))
        if not split:
            groups.append((idx, table_z, "plain"))
    comps = []
    for idx, support_z, kind in groups:
        p = obj_xyz[idx]
        c = obj_rgb[idx]
        if len(p) < MIN_CLUSTER_PTS:
            continue
        ext = p.max(axis=0) - p.min(axis=0)
        h_top = float(np.percentile(p[:, 2], 95) - table_z)
        top = p[p[:, 2] > table_z + h_top - 0.008]
        tc = c[p[:, 2] > table_z + h_top - 0.008]
        center, (w, l), ang = rect_xy(top[:, :2] if len(top) >= 3 else p[:, :2])
        fcenter, (fw, fl), fang = rect_xy(p[:, :2])
        hsv = to_hsv(c)
        hsv_top = to_hsv(tc) if len(tc) else hsv
        comp = {
            "n": int(len(idx)),
            "center_xy": center,
            "foot_center_xy": fcenter,
            "top_z": float(table_z + h_top),
            "height": h_top,
            "top_wl": (float(w), float(l)),
            "foot_wl": (float(fw), float(fl)),
            "yaw": float(ang),
            "extent": ext.tolist(),
            "hsv_med": np.median(hsv_top, axis=0).tolist(),
            "color": color_name(np.median(hsv_top, axis=0)),
            "hsv": hsv,
            "hsv_top": hsv_top,
            "zmax": float(p[:, 2].max()),
            "support_z": float(support_z),
            "kind": kind,
            "size": float(table_z + h_top - support_z),
        }
        comp["is_big"] = bool(max(ext[0], ext[1]) > MAX_OBJ_EXTENT or comp["zmax"] - table_z > 0.15)
        comps.append(comp)
    # flat / low coloured points (target candidates; a disk may be up to ~1.6 cm thick)
    flat = (rel > -FLAT_BAND) & (rel < TARGET_BAND_MAX)
    f_xyz, f_rgb = xyz[flat], rgb[flat]
    flat_info = {"xyz": f_xyz, "rgb": f_rgb, "table_z": table_z}
    if len(f_rgb):
        hsv = to_hsv(f_rgb)
        flat_info["hsv"] = hsv
        tf = np.abs(f_xyz[:, 2] - table_z) < 0.003
        flat_info["table_hsv"] = np.median(hsv[tf] if tf.sum() > 20 else hsv, axis=0).tolist()
    return comps, flat_info


def find_target(flat_info, prefer="red", exclude_xy=(), log=None):
    """Flat target disk: enclosing circle of the largest flat cluster of `prefer`
    coloured points, falling back to any saturated flat cluster."""
    if "hsv" not in flat_info or len(flat_info["xyz"]) == 0:
        return None
    xyz, hsv = flat_info["xyz"], flat_info["hsv"]
    th = flat_info["table_hsv"]
    results = []
    for mode in ("prefer", "salient"):
        if mode == "prefer":
            name = COLOR_ALIASES.get(prefer, prefer)
            m = np.zeros(len(hsv), bool)
            for lo, hi in COLOR_HUES.get(name, []):
                m |= (hsv[:, 0] >= lo) & (hsv[:, 0] < hi)
            m &= (hsv[:, 1] > 90) & (hsv[:, 2] > 50)
            dht = np.abs(hsv[:, 0] - th[0])
            dht = np.minimum(dht, 180 - dht)
            m &= (dht > 6) | (hsv[:, 1] > th[1] + 60)
        else:
            dh = np.abs(hsv[:, 0] - th[0])
            dh = np.minimum(dh, 180 - dh)
            m = (hsv[:, 1] > 100) & ((dh > 12) | (np.abs(hsv[:, 1] - th[1]) > 80))
        if m.sum() < 15:
            continue
        zc = xyz[m, 2]
        # dominant surface height of these points (a disk is planar, cube sides are not)
        bins = np.arange(zc.min() - 0.001, zc.max() + 0.003, 0.002)
        if len(bins) >= 3:
            hh, ee = np.histogram(zc, bins=bins)
            zi = int(np.argmax(hh))
            zmode = 0.5 * (ee[zi] + ee[zi + 1])
        else:
            zmode = float(np.median(zc))
        mm = np.zeros(len(m), bool)
        mm[np.nonzero(m)[0][np.abs(zc - zmode) < 0.003]] = True
        m = mm
        if m.sum() < 15:
            continue
        p, _ = voxel_down(xyz[m], hsv[m], vox=0.003)
        cl = clusters(p, r=0.02, min_pts=8)
        best = None
        for idx in cl:
            q = p[idx]
            (cx, cy), rad = cv2.minEnclosingCircle((q[:, :2] * 1000).astype(np.float32))
            rad /= 1000.0
            if rad < 0.015 or rad > 0.2:
                continue
            score = len(idx)
            if best is None or score > best[0]:
                best = (score, np.array([cx, cy]) / 1000.0, rad, len(idx))
        if best is not None:
            results.append({"mode": mode, "center_xy": best[1], "radius": best[2], "n": best[3],
                            "thick": max(0.0, float(zmode - flat_info.get("table_z", zmode)))})
            break
    return results[0] if results else None
