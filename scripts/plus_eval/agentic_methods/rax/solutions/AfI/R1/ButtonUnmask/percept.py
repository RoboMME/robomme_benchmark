"""Perception: calibration self-check, point clouds, height maps, colour blobs, objects."""
import math
import re

import numpy as np

from geom import CONVENTIONS, label, min_area_rect, rgb_to_hsv
from obs import reply_meta, walk

# ------------------------------------------------------------------ colours
COLOR_WORDS = {
    "red": "red", "crimson": "red", "orange": "orange", "yellow": "yellow", "gold": "yellow",
    "green": "green", "lime": "green", "cyan": "cyan", "teal": "cyan", "turquoise": "cyan",
    "blue": "blue", "navy": "blue", "purple": "purple", "violet": "purple",
    "magenta": "magenta", "pink": "magenta", "brown": "brown", "white": "white",
    "black": "black", "gray": "gray", "grey": "gray",
}
HUE_CENTERS = {"red": 0.0, "orange": 28.0, "yellow": 58.0, "green": 120.0, "cyan": 180.0,
               "blue": 225.0, "purple": 272.0, "magenta": 315.0}
CHROMATIC = list(HUE_CENTERS)


def hue_class(h):
    """Map hue degrees array to index into CHROMATIC (nearest centre)."""
    centers = np.array([HUE_CENTERS[c] for c in CHROMATIC])
    d = np.abs(((h[..., None] - centers) + 180) % 360 - 180)
    return np.argmin(d, -1)


def parse_targets(instruction):
    """Ordered colour targets for the cubes named in the instruction."""
    s = (instruction or "").lower()
    toks = re.findall(r"[a-z]+", s)
    hits = []
    for i, t in enumerate(toks):
        if t in COLOR_WORDS:
            nxt = toks[i + 1:i + 3]
            hits.append((i, COLOR_WORDS[t], nxt))
    cube_words = {"cube", "cubes", "block", "blocks", "object", "ball", "item", "one"}
    holder_words = {"button", "buttons", "container", "containers", "bin", "bins", "box", "boxes",
                    "cup", "cups", "bowl", "bowls", "lid", "cover", "mask"}
    cubes = []
    for i, c, _ in hits:
        for t in toks[i + 1:i + 6]:
            if t in cube_words:
                cubes.append(c)
                break
            if t in holder_words:
                break
    if not cubes:
        cubes = [c for i, c, nxt in hits if not any(w in holder_words for w in nxt)]
    out = []
    for c in cubes:
        if not out or out[-1] != c:
            out.append(c)
    return out


# ------------------------------------------------------------------ projection
def _vec3(d):
    if isinstance(d, dict) and all(k in d for k in ("x", "y", "z")):
        try:
            return np.array([float(d["x"]), float(d["y"]), float(d["z"])])
        except Exception:
            return None
    if isinstance(d, dict):
        for k in ("position", "world_position", "position_m", "world", "point", "xyz", "world_point",
                  "point_world", "world_xyz", "p"):
            v = d.get(k)
            if isinstance(v, list) and len(v) == 3 and all(isinstance(a, (int, float)) for a in v):
                return np.array(v, float)
            if isinstance(v, dict):
                r = _vec3(v)
                if r is not None:
                    return r
        for k in ("world", "world_frame", "result"):
            if isinstance(d.get(k), dict):
                r = _vec3(d[k])
                if r is not None:
                    return r
        for k, v in d.items():
            if isinstance(v, list) and len(v) == 3 and all(isinstance(a, (int, float)) for a in v) \
                    and any(s in str(k).lower() for s in ("pos", "point", "xyz", "world")):
                return np.array(v, float)
    return None


def parse_pixel_reply(reply, n):
    meta = reply_meta(reply)
    lists = []
    for p, v in walk(meta):
        if isinstance(v, list) and len(v) == n and all(isinstance(x, dict) for x in v):
            lists.append((len(p), v))
    lists.sort(key=lambda a: a[0])
    out = [None] * n
    if lists:
        for i, d in enumerate(lists[0][1]):
            if d.get("valid") is False:
                continue
            out[i] = _vec3(d)
    elif n == 1:
        out[0] = _vec3(meta)
    return out


class Projector:
    """Back-projection for one observation; local camera model when validated, else MCP tool."""

    def __init__(self, ctx, obs, calib, log):
        self.ctx, self.obs, self.calib, self.log = ctx, obs, calib, log
        self.tool_calls = 0

    def mode(self, cam):
        c = self.calib.get(cam)
        if c and c.get("conv") is not None and cam in self.obs.cams and cam in self.obs.depth:
            return "local"
        if c and c.get("tool_ok"):
            return "tool"
        return None

    def points(self, cam, us, vs):
        us = np.asarray(us, float)
        vs = np.asarray(vs, float)
        m = self.mode(cam)
        P = np.full((len(us), 3), np.nan)
        if m == "local":
            d = self.obs.depth[cam]
            ui = np.clip(us.astype(int), 0, d.shape[1] - 1)
            vi = np.clip(vs.astype(int), 0, d.shape[0] - 1)
            dd = d[vi, ui]
            ok = dd > 1e-4
            if ok.any():
                P[ok] = self.obs.cams[cam].backproject(us[ok], vs[ok], dd[ok], self.calib[cam]["conv"])
            return P
        if m == "tool":
            return tool_points(self.ctx, self.obs.step_id, cam, us, vs, self)
        return P


def tool_points(ctx, step_id, cam, us, vs, counter=None):
    P = np.full((len(us), 3), np.nan)
    for s in range(0, len(us), 32):
        sel = [{"camera": cam, "u": float(u), "v": float(v)} for u, v in zip(us[s:s + 32], vs[s:s + 32])]
        try:
            r = ctx.call("robomme_pixel_to_position", observation_step_id=int(step_id or 0), selections=sel,
                         depth_window_radius=0)
            if counter is not None:
                counter.tool_calls += 1
            pts = parse_pixel_reply(r, len(sel))
            for i, p in enumerate(pts):
                if p is not None:
                    P[s + i] = p
        except Exception:
            pass
    return P


def calibrate(ctx, obs, log):
    """Pick the back-projection convention that reproduces robomme_pixel_to_position."""
    calib = {}
    for cam in ("front", "wrist"):
        info = {"conv": None, "tool_ok": False, "err": None}
        calib[cam] = info
        H = W = 256
        if cam in obs.depth:
            H, W = obs.depth[cam].shape
        # sample pixels on a grid with valid, locally smooth depth
        us, vs = [], []
        for v in np.linspace(H * 0.12, H * 0.88, 5):
            for u in np.linspace(W * 0.12, W * 0.88, 5):
                us.append(int(u))
                vs.append(int(v))
        us = np.array(us)
        vs = np.array(vs)
        if cam in obs.depth:
            d = obs.depth[cam]
            ok = d[vs, us] > 1e-4
            us, vs = us[ok], vs[ok]
        if len(us) == 0:
            log(f"calib {cam}: no depth samples")
            continue
        try:
            Pt = tool_points(ctx, obs.step_id, cam, us, vs)
        except Exception as e:
            log(f"calib {cam}: tool error {e}")
            Pt = np.full((len(us), 3), np.nan)
        okt = np.isfinite(Pt).all(1)
        info["tool_ok"] = bool(okt.sum() >= 3)
        if cam not in obs.cams or cam not in obs.depth or okt.sum() < 3:
            log(f"calib {cam}: cams={cam in obs.cams} depth={cam in obs.depth} tool_valid={int(okt.sum())}")
            if cam in obs.cams and cam in obs.depth and okt.sum() < 3:
                info["conv"] = ("gl", 0.0, "z", False, "")  # documented convention, unverified
                info["err"] = None
                log(f"calib {cam}: tool unavailable, using documented OpenGL convention")
            continue
        d = obs.depth[cam][vs, us]
        best = None
        for conv in CONVENTIONS:
            try:
                Pm = obs.cams[cam].backproject(us + 0.0, vs + 0.0, d, conv)
                e = np.linalg.norm(Pm[okt] - Pt[okt], axis=1)
                med = float(np.median(e))
                if best is None or med < best[0]:
                    best = (med, conv)
            except Exception:
                continue
        if best is not None:
            info["err"] = best[0]
            if best[0] < 0.015:
                info["conv"] = best[1]
        log(f"calib {cam}: best={best} n={int(okt.sum())} mode={'local' if info['conv'] else 'tool'}")
    return calib


# ------------------------------------------------------------------ clouds
def cloud(proj, cam, stride=1, tcp=None, table_z=None, excl=True):
    """World points for one camera: returns P (N,3), uv (N,2), rgb (N,3)."""
    obs = proj.obs
    rgb = obs.rgb.get(cam)
    m = proj.mode(cam)
    if m is None:
        return np.zeros((0, 3)), np.zeros((0, 2), int), np.zeros((0, 3), np.uint8)
    if m == "local":
        d = obs.depth[cam]
        H, W = d.shape
    else:
        H, W = (rgb.shape[:2] if rgb is not None else (256, 256))
        stride = max(stride, 3)
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    uu = uu.ravel()
    vv = vv.ravel()
    if m == "local":
        ok = obs.depth[cam][vv, uu] > 1e-4
        uu, vv = uu[ok], vv[ok]
    P = proj.points(cam, uu, vv)
    ok = np.isfinite(P).all(1)
    P, uu, vv = P[ok], uu[ok], vv[ok]
    if rgb is not None and rgb.shape[0] == H and rgb.shape[1] == W:
        C = rgb[vv, uu]
    elif rgb is not None:
        C = rgb[np.clip(vv * rgb.shape[0] // H, 0, rgb.shape[0] - 1), np.clip(uu * rgb.shape[1] // W, 0, rgb.shape[1] - 1)]
    else:
        C = np.full((len(uu), 3), 128, np.uint8)
    keep = np.ones(len(P), bool)
    if excl and tcp is not None:
        keep &= ~robot_mask(P, tcp)
        if cam == "wrist":
            camp = obs.cams[cam].position() if cam in obs.cams else None
            if camp is not None:
                keep &= np.linalg.norm(P - camp, axis=1) > 0.06
    return P[keep], np.stack([uu, vv], 1)[keep], C[keep]


def robot_mask(P, tcp, r=0.11, below=0.035):
    dxy = np.linalg.norm(P[:, :2] - tcp[None, :2], axis=1)
    return (dxy < r) & (P[:, 2] > tcp[2] - below)


def estimate_table(P):
    z = P[:, 2]
    z = z[np.isfinite(z) & (np.abs(z) < 2.0)]
    if len(z) < 50:
        return 0.0
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.003))
    i = int(np.argmax(hist))
    sel = z[(z >= edges[i] - 0.003) & (z <= edges[i + 1] + 0.003)]
    return float(np.median(sel))


# ------------------------------------------------------------------ height map
class HeightMap:
    def __init__(self, P, C, table_z, cell=0.005, zmax=0.30, bounds=None, fill_holes=False):
        self.t = table_z
        self.cell = cell
        sel = np.isfinite(P).all(1) & (P[:, 2] < table_z + zmax) & (P[:, 2] > table_z - 0.05)
        P, C = P[sel], C[sel]
        if bounds is None:
            near = P[np.abs(P[:, 2] - table_z) < 0.02]
            if len(near) < 20:
                near = P
            lo = np.percentile(near[:, :2], 0.5, axis=0) - 0.05
            hi = np.percentile(near[:, :2], 99.5, axis=0) + 0.05
            lo = np.maximum(lo, near[:, :2].mean(0) - 1.2)
            hi = np.minimum(hi, near[:, :2].mean(0) + 1.2)
        else:
            lo, hi = np.array(bounds[0]), np.array(bounds[1])
        self.lo = lo
        self.shape = (max(1, int(math.ceil((hi[0] - lo[0]) / cell))), max(1, int(math.ceil((hi[1] - lo[1]) / cell))))
        ij = np.floor((P[:, :2] - lo) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] < self.shape[1])
        ij, P, C = ij[ok], P[ok], C[ok]
        self.hz = np.full(self.shape, -np.inf)
        self.col = np.zeros(self.shape + (3,), np.uint8)
        order = np.argsort(P[:, 2])
        ij, P, C = ij[order], P[order], C[order]
        self.hz[ij[:, 0], ij[:, 1]] = P[:, 2]  # last (highest) wins
        self.col[ij[:, 0], ij[:, 1]] = C
        gi, gj = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        self.pxy = self.lo + (np.stack([gi, gj], -1) + 0.5) * cell
        self.pxy[ij[:, 0], ij[:, 1]] = P[:, :2]
        self.n = np.zeros(self.shape, int)
        np.add.at(self.n, (ij[:, 0], ij[:, 1]), 1)
        if fill_holes:
            try:
                from scipy import ndimage
                hz0 = np.where(np.isfinite(self.hz), self.hz, -10.0)
                mf = ndimage.maximum_filter(hz0, size=3)
                empty = ~np.isfinite(self.hz) & (mf > -10.0)
                if empty.any():
                    # colour from the neighbour that supplied the max
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            sh = np.roll(np.roll(hz0, di, 0), dj, 1)
                            shc = np.roll(np.roll(self.col, di, 0), dj, 1)
                            m = empty & (sh == mf)
                            self.col[m] = shc[m]
                    self.hz[empty] = mf[empty]
            except Exception:
                pass

    def xy_of(self, i, j):
        return self.lo + (np.stack([i, j], -1) + 0.5) * self.cell

    def ij_of(self, xy):
        return np.floor((np.asarray(xy) - self.lo) / self.cell).astype(int)

    def objects(self, min_h=0.008, min_cells=None, tall=0.16):
        if min_cells is None:
            min_cells = max(3, int(round(1.25e-4 / self.cell ** 2)))
        mask = self.hz > self.t + min_h
        lab, n = label(mask)
        objs = []
        for k in range(1, n + 1):
            ii, jj = np.nonzero(lab == k)
            if len(ii) < min_cells:
                continue
            z = self.hz[ii, jj]
            top = float(np.percentile(z, 95))
            h = top - self.t
            xy = self.xy_of(ii, jj)
            upper = z > self.t + 0.5 * h
            c, (L, W), ang = min_area_rect(xy[upper] if upper.sum() >= 3 else xy)
            L += self.cell
            W += self.cell
            toplayer = z >= top - max(0.006, 0.15 * h)
            tc, (tL, tW), _ = min_area_rect(xy[toplayer] if toplayer.sum() >= 3 else xy)
            cols = self.col[ii, jj][toplayer]
            hh, ss, vv = rgb_to_hsv(cols[None])
            ss, vv, hh = ss[0], vv[0], hh[0]
            sat = ss > 0.4
            dom = None
            frac = 0.0
            if sat.sum() > 0:
                cls = hue_class(hh[sat])
                cnt = np.bincount(cls, minlength=len(CHROMATIC))
                dom = CHROMATIC[int(np.argmax(cnt))]
                frac = float(cnt.max()) / max(1, len(cols))
            fill = len(ii) * self.cell ** 2 / max(1e-6, L * W)
            objs.append(dict(id=k, n=len(ii), center=c, top_center=tc, top=top, h=h, L=L, W=W, ang=ang,
                             tall=bool(top > self.t + tall), cells=xy, dom=dom, domfrac=frac,
                             mean_rgb=cols.mean(0).tolist() if len(cols) else [0, 0, 0],
                             sat=float(ss.mean()) if len(ss) else 0.0, fill=fill,
                             toparea=float(toplayer.sum()) * self.cell ** 2, tL=tL, tW=tW))
        return objs


def top_face(hm, xy_hint, radius=0.05, min_h=0.015, tol=0.007):
    """Flat top surface of the object under/near xy_hint (connected, no closing)."""
    xy_hint = np.asarray(xy_hint, float)
    ii, jj = np.nonzero(np.isfinite(hm.hz) & (hm.hz > hm.t + min_h))
    if len(ii) == 0:
        return None
    xy = hm.xy_of(ii, jj)
    dd = np.linalg.norm(xy - xy_hint[None], axis=1)
    near = dd < 0.03
    if near.sum() < 2:
        near = dd < radius
        if near.sum() < 2:
            return None
    top = float(np.percentile(hm.hz[ii[near], jj[near]], 90))
    mask = np.isfinite(hm.hz) & (np.abs(hm.hz - top) < tol) & (hm.hz > hm.t + min_h)
    full_mask = mask
    try:
        from scipy import ndimage
        er = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
        if er.sum() >= 4:
            mask = er
    except Exception:
        pass
    lab, n = label(mask)
    if n == 0:
        return None
    li, lj = np.nonzero(lab > 0)
    lxy = hm.xy_of(li, lj)
    d = np.linalg.norm(lxy - xy_hint[None], axis=1)
    k = int(np.argmin(d))
    if d[k] > radius:
        return None
    comp = lab[li[k], lj[k]]
    region = lab == comp
    if mask is not full_mask:
        try:
            from scipy import ndimage
            region = ndimage.binary_dilation(region, structure=np.ones((3, 3))) & full_mask
        except Exception:
            pass
    ci, cj = np.nonzero(region)
    cxy = hm.pxy[ci, cj] if hasattr(hm, "pxy") else hm.xy_of(ci, cj)
    c, (L, W), ang = min_area_rect(cxy)
    L += hm.cell
    W += hm.cell
    return dict(center=c, L=L, W=W, ang=ang, top=float(np.median(hm.hz[ci, cj])), n=int(len(ci)),
                area=float(len(ci)) * hm.cell ** 2, dist=float(d[k]))


def obj_brief(o):
    return dict(id=o["id"], c=[round(float(a), 3) for a in o["center"]], top=round(o["top"], 3),
                h=round(o["h"], 3), L=round(o["L"], 3), W=round(o["W"], 3), ang=round(o["ang"], 2),
                tall=o["tall"], dom=o["dom"], domfrac=round(o["domfrac"], 2), n=o["n"],
                rgb=[int(a) for a in o["mean_rgb"]], fill=round(o["fill"], 2))


def dist_to_obj(o, xy):
    return float(np.min(np.linalg.norm(o["cells"] - np.asarray(xy)[None], axis=1)))


# ------------------------------------------------------------------ colour blobs
def color_blobs(proj, cam, table_z, tcp, min_px=6, sat_th=0.42, val_th=0.15):
    """Saturated colour blobs above the table: list of dict(color, xy, ztop, npx, ext)."""
    obs = proj.obs
    rgb = obs.rgb.get(cam)
    if rgb is None or proj.mode(cam) is None:
        return []
    h, s, v = rgb_to_hsv(rgb)
    mask = (s > sat_th) & (v > val_th)
    if cam in obs.depth and obs.depth[cam].shape == mask.shape:
        mask &= obs.depth[cam] > 1e-4
    if mask.sum() == 0:
        return []
    cls = hue_class(h)
    out = []
    for ci, cname in enumerate(CHROMATIC):
        m = mask & (cls == ci)
        if m.sum() < min_px:
            continue
        lab, n = label(m)
        for k in range(1, n + 1):
            vv, uu = np.nonzero(lab == k)
            if len(vv) < min_px:
                continue
            if proj.mode(cam) == "tool" and len(vv) > 12:
                idx = np.linspace(0, len(vv) - 1, 12).astype(int)
                vs_, us_ = vv[idx], uu[idx]
            else:
                vs_, us_ = vv, uu
            P = proj.points(cam, us_, vs_)
            P = P[np.isfinite(P).all(1)]
            if len(P) < 3:
                continue
            P = P[P[:, 2] > table_z + 0.004]
            if tcp is not None and len(P):
                P = P[~robot_mask(P, tcp, r=0.07, below=0.03)]
            if len(P) < 3:
                continue
            med = np.median(P, 0)
            zt = float(np.percentile(P[:, 2], 95))
            topP = P[P[:, 2] >= zt - 0.006]
            if len(topP) >= 3:
                med[:2] = np.median(topP[:, :2], 0)
            ext = float(np.max(np.ptp(P[:, :2], axis=0)))
            out.append(dict(color=cname, xy=med[:2].copy(), ztop=float(np.percentile(P[:, 2], 95)),
                            npx=int(len(vv)), ext=ext, cam=cam, uv=(float(uu.mean()), float(vv.mean()))))
    return out


class Tracker:
    """Last confident world position of each colour cube."""

    def __init__(self, table_z):
        self.t = table_z
        self.sight = {}  # colour -> list of sightings
        self.exclude = []  # (xy, radius) e.g. button

    def add(self, blobs, step, source):
        for b in blobs:
            if b["ext"] > 0.09 or b["ztop"] > self.t + 0.16:
                continue
            if any(np.linalg.norm(b["xy"] - np.asarray(xy)) < r for xy, r in self.exclude):
                continue
            rec = dict(b)
            rec["step"] = step
            rec["source"] = source
            self.sight.setdefault(b["color"], []).append(rec)

    def last(self, color, source=None):
        L = [s for s in self.sight.get(color, []) if source is None or s["source"] == source]
        if not L:
            return None
        # group by spatial cluster; take the largest-npx cluster, then its latest strong sighting
        mx = {}
        for s in L:
            mx[s["cam"]] = max(mx.get(s["cam"], 0), s["npx"])
        strong = [s for s in L if s["npx"] >= 0.35 * mx[s["cam"]]]
        anchor = max(strong, key=lambda s: s["npx"] / mx[s["cam"]] + 0.01 * s["step"])
        near = [s for s in strong if np.linalg.norm(s["xy"] - anchor["xy"]) < 0.05]
        latest = max(near, key=lambda s: (s["step"], s["npx"]))
        xs = [s["xy"] for s in near if s["step"] == latest["step"]]
        out = dict(latest)
        out["xy"] = np.mean(xs, 0)
        out["n_sight"] = len(L)
        return out

    def visible_now(self, color, step):
        return any(s["step"] == step for s in self.sight.get(color, []))
