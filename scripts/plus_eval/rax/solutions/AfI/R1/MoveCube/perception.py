"""Observation assembly, calibration, table/object/target perception (no Gemini)."""
import math
import time
from collections import deque

import numpy as np

import geom
import mcio

LOG = print


class Obs:
    def __init__(self, reply, ctx=None):
        self.reply = reply
        self.meta = mcio.meta(reply)
        self.step_id = mcio.get_step_id(self.meta)
        self.state = mcio.get_state(self.meta)
        self.images = {}
        self.src = {}
        self._load()
        self.K, self.pose = {}, {}
        for cam in ("front", "wrist"):
            K, pose = geom.camera_params(self.meta, cam)
            if K:
                self.K[cam] = K
            if pose:
                self.pose[cam] = pose

    def _load(self):
        # 1) explicit file records
        recs = mcio.file_records(self.meta)
        if recs:
            fake = {"files": recs}
            try:
                imgs = mcio.frame_images(fake)
                for c, d in imgs.items():
                    for k, a in d.items():
                        self.images.setdefault(c, {}).setdefault(k, a)
                        self.src[(c, k)] = "files"
            except Exception as e:
                LOG("obs file records failed", repr(e))
        # 2) frame_sequence manifest final frame (if it is this step)
        try:
            man = mcio.read_manifest(self.meta)
            if man and man.get("frames"):
                frames = sorted(man["frames"], key=lambda f: f.get("frame_index", 0))
                last = frames[-1]
                ok = (self.step_id is None or last.get("step_id") is None
                      or last.get("step_id") == self.step_id)
                if ok and last.get("phase", "execution") != "video_demo":
                    imgs = mcio.frame_images(last)
                    for c, d in imgs.items():
                        for k, a in d.items():
                            if k not in self.images.get(c, {}):
                                self.images.setdefault(c, {})[k] = a
                                self.src[(c, k)] = "manifest"
        except Exception as e:
            LOG("obs manifest failed", repr(e))
        # 3) inline attachments
        try:
            imgs = mcio.inline_images(self.reply)
            for c, d in imgs.items():
                for k, a in d.items():
                    if k not in self.images.get(c, {}):
                        self.images.setdefault(c, {})[k] = a
                        self.src[(c, k)] = "inline"
        except Exception as e:
            LOG("obs inline failed", repr(e))

    def rgb(self, cam):
        return self.images.get(cam, {}).get("rgb")

    def depth(self, cam):
        d = self.images.get(cam, {}).get("depth")
        if d is not None and d.ndim == 2 and np.count_nonzero(d) > 100:
            return d
        return None

    def summary(self):
        return {"step": self.step_id, "state": None if self.state is None else [round(x, 4) for x in self.state],
                "imgs": {c: {k: list(a.shape) for k, a in d.items()} for c, d in self.images.items()},
                "src": {f"{c}/{k}": s for (c, k), s in self.src.items()},
                "K": list(self.K.keys()), "pose": list(self.pose.keys())}


def parse_p2p(reply, n):
    """Parse robomme_pixel_to_position reply -> list of (xyz or None)."""
    m = mcio.meta(reply)
    best = None
    for p, node in mcio.nodes(m):
        if isinstance(node, list) and len(node) == n and all(isinstance(x, dict) for x in node):
            best = node
            break
    out = []
    if best is None:
        return [None] * n
    for d in best:
        if d.get("valid") is False:
            out.append(None)
            continue
        val = None
        for k, v in d.items():
            kl = k.lower()
            if any(s in kl for s in ("world", "pos", "point", "xyz")) and isinstance(v, list) and len(v) == 3 \
                    and all(isinstance(x, (int, float)) for x in v):
                val = np.array(v, float)
                break
            if isinstance(v, dict):
                if all(a in v for a in ("x", "y", "z")) and any(sx in kl for sx in ("world", "pos", "point", "xyz")):
                    try:
                        val = np.array([float(v["x"]), float(v["y"]), float(v["z"])])
                        break
                    except Exception:
                        pass
                for k2, v2 in v.items():
                    if isinstance(v2, list) and len(v2) == 3 and all(isinstance(x, (int, float)) for x in v2):
                        val = np.array(v2, float)
                        break
                if val is not None:
                    break
        if val is None:
            for k, v in d.items():
                if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                    val = np.array(v, float)
                    break
        out.append(val)
    return out


class Cloud:
    """Grid of world points for one camera image."""

    def __init__(self, xyz, valid, rgb, uv, cam):
        self.xyz, self.valid, self.rgb, self.uv, self.cam = xyz, valid, rgb, uv, cam
        self.shape = valid.shape


class Perceiver:
    def __init__(self, ctx, log=print):
        self.ctx = ctx
        self.log = log
        self.variant = {}       # cam -> variant or 'tool'
        self.table_z = None
        self.calib_err = {}

    # ---------------- calibration ----------------
    def tool_points(self, step_id, cam, uvs, radius=0):
        out = []
        for i in range(0, len(uvs), 32):
            chunk = uvs[i:i + 32]
            sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in chunk]
            try:
                r = self.ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id or 0),
                                  selections=sel, depth_window_radius=int(radius))
                out.extend(parse_p2p(r, len(chunk)))
            except Exception as e:
                self.log("p2p failed", repr(e)[:300])
                out.extend([None] * len(chunk))
        return out

    def calibrate(self, obs, cam):
        if cam in self.variant:
            return self.variant[cam]
        d = obs.depth(cam)
        K, pose = obs.K.get(cam), obs.pose.get(cam)
        H, W = (256, 256) if d is None else d.shape
        rng = np.random.default_rng(0)
        uvs = []
        if d is not None:
            vv, uu = np.nonzero(d > 0)
            if len(uu):
                idx = rng.choice(len(uu), size=min(12, len(uu)), replace=False)
                uvs = [(int(uu[i]), int(vv[i])) for i in idx]
        if not uvs:
            uvs = [(int(u), int(v)) for u in (64, 128, 192) for v in (96, 160, 220)]
        tp = self.tool_points(obs.step_id, cam, uvs)
        pairs = [(uv, p) for uv, p in zip(uvs, tp) if p is not None]
        self.log(f"calib {cam}: {len(pairs)} tool pts, K={K is not None} pose={pose is not None} depth={d is not None}")
        if d is None or K is None or pose is None:
            self.variant[cam] = "tool" if pairs else None
            return self.variant[cam]
        if len(pairs) < 3:
            v = self.flatness_variant(d, K, pose) if cam == "front" else None
            if v is None and cam == "wrist" and "front" in self.variant and self.variant["front"] not in (None, "tool"):
                v = self.variant["front"]
            self.log(f"calib {cam}: tool unusable; variant by flatness/front = {v}")
            self.variant[cam] = v
            return v
        best, berr = None, 1e9
        for var in geom.VARIANTS:
            u = np.array([a[0][0] for a in pairs]); v = np.array([a[0][1] for a in pairs])
            dd = d[v, u]
            P = geom.backproject(u, v, dd, K, pose, var)
            err = np.median(np.linalg.norm(P - np.array([a[1] for a in pairs]), axis=1))
            if err < berr:
                best, berr = var, err
        self.calib_err[cam] = berr
        self.log(f"calib {cam}: best variant {best} median err {berr:.4f} m")
        self.variant[cam] = best if berr < 0.02 else "tool"
        return self.variant[cam]

    def flatness_variant(self, d, K, pose):
        """Choose the projection convention under which the scene has a large horizontal plane below the camera."""
        vv, uu = np.mgrid[0:d.shape[0]:4, 0:d.shape[1]:4]
        dd = d[vv, uu].ravel(); ok = dd > 0
        best, bs = None, -1
        for var in geom.VARIANTS:
            P = geom.backproject(uu.ravel()[ok], vv.ravel()[ok], dd[ok], K, pose, var)
            z = P[:, 2]
            hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
            if len(hist) == 0:
                continue
            i = int(np.argmax(hist)); frac = hist[i] / len(z)
            above = float(np.asarray(pose[1])[2] - edges[i]) > 0.1
            sc = frac + (0.5 if above else 0) + (0.05 if var == ("gl", 0.0, False) else 0)
            if sc > bs:
                best, bs = var, sc
        return best

    # ---------------- clouds ----------------
    def cloud(self, obs, cam, stride=1):
        var = self.calibrate(obs, cam)
        rgb = obs.rgb(cam)
        d = obs.depth(cam)
        if var not in (None, "tool") and d is not None:
            H, W = d.shape
            vv, uu = np.mgrid[0:H:stride, 0:W:stride]
            dd = d[vv, uu]
            xyz = geom.backproject(uu.ravel(), vv.ravel(), dd.ravel(), obs.K[cam], obs.pose[cam], var).reshape(
                vv.shape + (3,))
            valid = dd > 0
        elif var == "tool":
            s = max(stride, 4)
            H = W = 256 if rgb is None else rgb.shape[0]
            vv, uu = np.mgrid[0:H:s, 0:W:s]
            pts = self.tool_points(obs.step_id, cam, list(zip(uu.ravel(), vv.ravel())))
            xyz = np.zeros(uu.shape + (3,))
            valid = np.zeros(uu.shape, bool)
            for i, p in enumerate(pts):
                if p is not None:
                    xyz.flat[i * 3:i * 3 + 3] = p
                    valid.flat[i] = True
        else:
            return None
        if rgb is not None:
            col = rgb[np.clip(vv, 0, rgb.shape[0] - 1), np.clip(uu, 0, rgb.shape[1] - 1)].astype(float)
        else:
            col = np.full(vv.shape + (3,), 128.0)
        return Cloud(xyz, valid, col, np.stack([uu, vv], -1), cam)

    # ---------------- table ----------------
    def fit_table(self, cl):
        z = cl.xyz[..., 2][cl.valid]
        z = z[(z > -1.0) & (z < 1.0)]
        if len(z) < 100:
            return None
        bins = np.arange(z.min(), z.max() + 0.005, 0.005)
        if len(bins) < 3:
            return float(np.median(z))
        hist, edges = np.histogram(z, bins)
        thr = 0.15 * hist.max()
        # highest significant peak (table above floor)
        peaks = [i for i in range(len(hist)) if hist[i] >= thr and hist[i] >= hist[max(i - 1, 0)]
                 and hist[i] >= hist[min(i + 1, len(hist) - 1)]]
        # choose peak with largest count among those with count >= 0.5*max, prefer higher z
        cands = [i for i in peaks if hist[i] >= 0.5 * hist.max()] or [int(np.argmax(hist))]
        i = max(cands, key=lambda j: edges[j])
        z0 = 0.5 * (edges[i] + edges[i + 1])
        near = z[np.abs(z - z0) < 0.008]
        tz = float(np.median(near)) if len(near) else float(z0)
        return tz

    # ---------------- segmentation ----------------
    def components(self, cl, table_z, hmin=0.006, hmax=0.6, link=0.02, exclude=None):
        h = cl.xyz[..., 2] - table_z
        mask = cl.valid & (h > hmin) & (h < hmax)
        if exclude is not None:
            mask &= ~exclude
        H, W = mask.shape
        lab = -np.ones((H, W), int)
        comps = []
        xyz = cl.xyz
        for v0, u0 in zip(*np.nonzero(mask)):
            if lab[v0, u0] >= 0:
                continue
            cid = len(comps)
            q = deque([(v0, u0)])
            lab[v0, u0] = cid
            members = []
            while q:
                v, u = q.popleft()
                members.append((v, u))
                p = xyz[v, u]
                for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = v + dv, u + du
                    if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                        if np.sum((xyz[a, b] - p) ** 2) < link * link:
                            lab[a, b] = cid
                            q.append((a, b))
            comps.append(members)
        out = []
        for cid, mem in enumerate(comps):
            if len(mem) < 6:
                continue
            idx = tuple(np.array(mem).T)
            out.append(describe(xyz[idx], cl.rgb[idx], table_z, pix=np.array(mem), cid=cid))
        return out, lab


def min_area_yaw(xy):
    """Yaw (mod 90 deg) of the min-area bounding rectangle; returns yaw, center, (ext_a, ext_b)."""
    best = None
    c0 = xy.mean(0)
    for a in np.radians(np.arange(0, 90, 2.0)):
        c, s = math.cos(a), math.sin(a)
        p = (xy - c0) @ np.array([[c, -s], [s, c]])
        lo = np.percentile(p, 3, axis=0); hi = np.percentile(p, 97, axis=0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = c0 + np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
            best = (area, a, ctr, hi - lo)
    return best[1], best[2], best[3]


def fit_square(xy, s, occluder_xy=None):
    """Centre/yaw of a square of known side s from visible top points; handles one-sided occlusion."""
    yaw, ctr, ext = min_area_yaw(xy)
    c, sn = math.cos(yaw), math.sin(yaw)
    Rm = np.array([[c, -sn], [sn, c]])
    p = (xy - ctr) @ Rm
    lo = np.percentile(p, 3, axis=0); hi = np.percentile(p, 97, axis=0)
    local = (lo + hi) / 2
    for i in range(2):
        if hi[i] - lo[i] < 0.8 * s and occluder_xy is not None:
            oc = (np.asarray(occluder_xy)[:2] - ctr) @ Rm
            if oc[i] > 0:   # occluder on + side: trust the low edge
                local[i] = lo[i] + s / 2 - 0.002
            else:
                local[i] = hi[i] - s / 2 + 0.002
    return float(yaw), ctr + Rm @ local, (hi - lo)


def describe(pts, cols, table_z, pix=None, cid=-1):
    h = pts[:, 2] - table_z
    hmax = float(np.percentile(h, 97))
    xy = pts[:, :2]
    c = xy.mean(0)
    if len(xy) >= 3:
        cov = np.cov((xy - c).T) + 1e-12 * np.eye(2)
        w, V = np.linalg.eigh(cov)
        ax = V[:, 1]
    else:
        ax = np.array([1.0, 0.0])
    perp = np.array([-ax[1], ax[0]])
    pa = (xy - c) @ ax
    pp = (xy - c) @ perp
    L = float(np.percentile(pa, 98) - np.percentile(pa, 2))
    Wd = float(np.percentile(pp, 98) - np.percentile(pp, 2))
    top = pts[h > hmax - 0.012]
    return {"cid": cid, "n": len(pts), "pts": pts, "cols": cols, "pix": pix,
            "center": c, "axis": ax, "L": L, "W": Wd, "hmax": hmax,
            "color": np.median(cols, 0), "top": top,
            "zmax": float(np.percentile(pts[:, 2], 97))}


def chroma(c):
    c = np.asarray(c, float)
    s = c.sum(-1, keepdims=True) + 1e-6
    return c / s


def color_dist(a, b):
    """Distance combining chromaticity and brightness (0..~1.5)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ch = np.linalg.norm(chroma(a) - chroma(b), axis=-1)
    br = np.abs(a.mean(-1) - b.mean(-1)) / 255.0
    return 2.0 * ch + 0.5 * br


def classify_objects(objs):
    """Return (cube, tools, others) among non-robot components."""
    cands = []
    for o in objs:
        if o["hmax"] > 0.15:
            continue
        cands.append(o)
    cubes, tools, others = [], [], []
    for o in cands:
        L, W, h = o["L"], o["W"], o["hmax"]
        if 0.018 <= L <= 0.085 and 0.012 <= W <= 0.085 and 0.015 <= h <= 0.09 and L / max(W, 1e-3) < 2.0 \
                and o["n"] >= 6:
            score = abs(L - 0.042) + abs(W - 0.04) + abs(h - 0.04) + 0.3 * abs(L / max(W, 1e-3) - 1)
            cubes.append((score, o))
        elif L > 0.07 and h < 0.10 and o["n"] >= 8:
            tools.append(o)
        else:
            others.append(o)
    cubes.sort(key=lambda t: t[0])
    tools.sort(key=lambda o: -o["n"])
    return [c[1] for c in cubes], tools, others


def find_target(cl, table_z, objs, avoid_xy=(), log=print):
    """Flat, colour-distinct disc on the table. Returns dict or None."""
    h = cl.xyz[..., 2] - table_z
    flat = cl.valid & (np.abs(h) < 0.007)
    if flat.sum() < 200:
        return None
    cols = cl.rgb[flat]
    ref = np.median(cols, 0)
    refb = ref.mean()
    dist = np.zeros(cl.shape)
    dist[flat] = color_dist(cl.rgb[flat], ref)
    bright = np.zeros(cl.shape)
    bright[flat] = cl.rgb[flat].mean(-1) / max(refb, 1)
    chd = np.zeros(cl.shape)
    chd[flat] = np.linalg.norm(chroma(cl.rgb[flat]) - chroma(ref), axis=-1)
    thr_ch = max(0.06, float(np.percentile(chd[flat], 80)) * 2.0)
    cand = flat & ((chd > thr_ch) | (bright > 1.3))
    # exclude pixels close to any object point (objects' sides and shadows-at-base)
    xy = cl.xyz[..., :2]
    vv, uu = np.nonzero(cand)
    if len(vv) and objs:
        allp = np.concatenate([o["pts"][::2, :2] for o in objs], 0)
        cxy = xy[vv, uu]
        near = np.zeros(len(vv), bool)
        for i in range(0, len(allp), 256):
            dd = np.linalg.norm(cxy[:, None, :] - allp[None, i:i + 256, :], axis=2).min(1)
            near |= dd < 0.012
        cand[vv[near], uu[near]] = False
    for a in avoid_xy:
        cand &= ~(np.linalg.norm(xy - np.asarray(a)[:2], axis=-1) < 0.06)
    if cand.sum() < 8:
        return None
    # components with spatial linkage
    H, W = cand.shape
    lab = -np.ones((H, W), int)
    comps = []
    for v0, u0 in zip(*np.nonzero(cand)):
        if lab[v0, u0] >= 0:
            continue
        q = deque([(v0, u0)]); lab[v0, u0] = len(comps); mem = []
        while q:
            v, u = q.popleft(); mem.append((v, u))
            for dv in (-2, -1, 0, 1, 2):
                for du in (-2, -1, 0, 1, 2):
                    a, b = v + dv, u + du
                    if 0 <= a < H and 0 <= b < W and cand[a, b] and lab[a, b] < 0:
                        lab[a, b] = len(comps); q.append((a, b))
        comps.append(mem)
    cands = []
    for mem in comps:
        if len(mem) < 8:
            continue
        idx = tuple(np.array(mem).T)
        p = cl.xyz[idx][:, :2]
        c = np.median(p, 0)
        ext = np.percentile(p, 97, 0) - np.percentile(p, 3, 0)
        r = 0.25 * (ext[0] + ext[1])
        if r < 0.012 or r > 0.2:
            continue
        aspect = max(ext) / max(min(ext), 1e-3)
        if aspect > 2.5:
            continue
        sat = float(np.mean(chd[idx]))
        score = len(mem) * (1.0 + 2 * sat) / aspect
        cands.append({"center": np.array([(np.percentile(p[:, 0], 97) + np.percentile(p[:, 0], 3)) / 2,
                                          (np.percentile(p[:, 1], 97) + np.percentile(p[:, 1], 3)) / 2]),
                      "radius": float(r), "n": len(mem), "score": score, "aspect": float(aspect),
                      "color": np.median(cl.rgb[idx], 0), "sat": sat})
    if not cands:
        return None
    cands.sort(key=lambda c: -c["score"])
    best = dict(cands[0])
    best["cands"] = [{"center": c["center"].tolist(), "radius": c["radius"], "score": c["score"]} for c in cands[:6]]
    return best


def color_track(cl, table_z, color, near=None, radius=0.25, thr=0.12, hmin=0.004, hmax=0.25,
                exclude_xy=None, exclude_r=0.0, exclude_zmin=None):
    """Points matching an object colour. Returns describe() or None."""
    h = cl.xyz[..., 2] - table_z
    m = cl.valid & (h > hmin) & (h < hmax)
    if near is not None:
        m &= np.linalg.norm(cl.xyz[..., :2] - np.asarray(near)[:2], axis=-1) < radius
    if exclude_xy is not None and exclude_r > 0:
        ex = np.linalg.norm(cl.xyz[..., :2] - np.asarray(exclude_xy)[:2], axis=-1) < exclude_r
        if exclude_zmin is not None:
            ex &= cl.xyz[..., 2] > exclude_zmin
        m &= ~ex
    if m.sum() < 4:
        return None
    d = color_dist(cl.rgb[m], color)
    sel = np.zeros_like(m)
    sel[m] = d < thr
    if sel.sum() < 4:
        return None
    pts = cl.xyz[sel]
    # keep the dominant spatial cluster (largest within 5 cm of the median)
    med = np.median(pts, 0)
    keep = np.linalg.norm(pts[:, :2] - med[:2], axis=1) < 0.05
    if keep.sum() < 4:
        keep = np.ones(len(pts), bool)
    return describe(pts[keep], cl.rgb[sel][keep], table_z)
