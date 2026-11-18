"""Detect the white target discs on the table in a front RGB-D frame (world xy)."""
import numpy as np
from PIL import Image
from geom import backproject


def _unused_load(path, depth=False):
    a = np.asarray(Image.open(path))
    return a.astype(float) * 0.001 if depth else a.convert("RGB") if False else np.asarray(Image.open(path).convert("RGB")).astype(int)


def detect_discs(rgb, depth_m, K, pose, zmax=0.03):
    rgb = rgb.astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(2), rgb.min(2)
    white = (mn > 150) & (mx - mn < 40)
    vv, uu = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    W = backproject(uu.astype(float), vv.astype(float), depth_m, K, pose)
    ok = white & (depth_m > 0) & (W[..., 2] < zmax) & (W[..., 2] > -0.02)
    # connected components on the image grid (4-neighbour), simple BFS
    lab = -np.ones(ok.shape, int)
    comps = []
    H, Wd = ok.shape
    for y0, x0 in zip(*np.nonzero(ok)):
        if lab[y0, x0] >= 0:
            continue
        stack = [(y0, x0)]
        lab[y0, x0] = len(comps)
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < Wd and ok[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = len(comps)
                    stack.append((ny, nx))
        comps.append(pix)
    discs = []
    for pix in comps:
        if len(pix) < 6:
            continue
        ys, xs = np.array(pix).T
        P = W[ys, xs]
        ext = P[:, :2].max(0) - P[:, :2].min(0)
        if ext.max() > 0.08:
            continue
        discs.append({"xy": P[:, :2].mean(0), "z": float(P[:, 2].mean()), "n": len(pix),
                      "uv": (float(xs.mean()), float(ys.mean())), "ext": ext})
    # merge components closer than 2.5 cm (a disc split by its printed rings)
    merged = []
    for d in sorted(discs, key=lambda d: -d["n"]):
        for m in merged:
            if np.linalg.norm(m["xy"] - d["xy"]) < 0.025:
                tot = m["n"] + d["n"]
                m["xy"] = (m["xy"] * m["n"] + d["xy"] * d["n"]) / tot
                m["n"] = tot
                break
        else:
            merged.append(dict(d))
    return merged


def filter_discs(ds, xlim=(-0.5, 0.3), min_n=15):
    return sorted([d for d in ds if xlim[0] < d["xy"][0] < xlim[1] and d["n"] >= min_n],
                  key=lambda d: d["xy"][1])


def similarity_from_pairs(a0, a1, b0, b1):
    """2-D similarity mapping a0->b0, a1->b1 (complex-number form)."""
    za0, za1, zb0, zb1 = (complex(*p) for p in (a0, a1, b0, b1))
    if abs(za1 - za0) < 1e-6:
        s = 1 + 0j
    else:
        s = (zb1 - zb0) / (za1 - za0)
    def f(p):
        z = complex(p[0], p[1])
        w = zb0 + s * (z - za0)
        return np.array([w.real, w.imag])
    return f


def find_anchors(path, vdiscs, radius=0.025):
    """Indices of disc visits along the path: (path_index, disc_index), by closest approach."""
    V = np.array([d["xy"] for d in vdiscs])
    D = np.linalg.norm(path[:, None, :2] - V[None], axis=2)
    near = D.min(1) < radius
    which = D.argmin(1)
    anchors = []
    i = 0
    n = len(path)
    while i < n:
        if near[i]:
            j = i
            while j + 1 < n and near[j + 1] and which[j + 1] == which[i]:
                j += 1
            k = i + int(np.argmin(D[i:j + 1, which[i]]))
            anchors.append((k, int(which[i])))
            i = j + 1
        else:
            i += 1
    return anchors


def map_path(path, vdiscs, edisc_xy, anchors):
    """Map each video path point into the execution layout, segment by segment."""
    V = [d["xy"] for d in vdiscs]
    E = list(edisc_xy)
    out = path.copy()
    if not anchors:
        return out
    # before first anchor: blend translation from 0 to the first disc's offset
    k0, d0 = anchors[0]
    off0 = E[d0] - V[d0]
    for i in range(0, k0 + 1):
        w = i / max(k0, 1)
        out[i, :2] = path[i, :2] + w * off0
    for (ka, da), (kb, db) in zip(anchors[:-1], anchors[1:]):
        if da != db:
            f = similarity_from_pairs(V[da], V[db], E[da], E[db])
        else:
            off = E[da] - V[da]
            f = lambda p, off=off: np.asarray(p[:2]) + off
        for i in range(ka, kb + 1):
            out[i, :2] = f(path[i, :2])
    kl, dl = anchors[-1]
    offl = E[dl] - V[dl]
    for i in range(kl, len(path)):
        out[i, :2] = path[i, :2] + offl
    return out
