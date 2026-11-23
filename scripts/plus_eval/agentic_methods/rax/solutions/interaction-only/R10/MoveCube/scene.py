"""Scene localization from front camera RGB-D."""
import numpy as np
from perc import *


def locate(meta, imgs, cam="front"):
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P, valid = world_points(meta, cam, dep)
    on = valid & (P[..., 2] > 0.004) & (P[..., 2] < 0.12) & (np.abs(P[..., 0]) < 1.0) & (np.abs(P[..., 1]) < 1.0)
    out = {}
    # cube
    best = None
    for c in components(mask_red(rgb) & on):
        if c.sum() < 4: continue
        if best is None or c.sum() > best.sum(): best = c
    if best is not None:
        pts = P[best]
        top = np.percentile(pts[:, 2], 95)
        tp = pts[pts[:, 2] > top - 0.006]
        c = np.median(tp, 0)
        out["cube"] = np.array([c[0], c[1], top])
        out["cube_px"] = int(best.sum())
    # target (flat, z small)
    tmask = mask_purple(rgb) & valid & (P[..., 2] < 0.03) & (np.abs(P[..., 0]) < 1.0)
    best = None
    for c in components(tmask):
        if c.sum() < 4: continue
        if best is None or c.sum() > best.sum(): best = c
    if best is not None:
        c0 = np.median(P[best], 0)
        allm = tmask & (np.linalg.norm(P[..., :2] - c0[:2], axis=-1) < 0.06)
        pts = P[allm]
        out["target"] = (pts.min(0) + pts.max(0)) / 2
        out["target_px"] = int(best.sum())
    # hook tool: teal handle component of small world extent near a grey shaft
    cands = []
    shafts = [c for c in components(mask_shaft(rgb) & on & (P[..., 2] < 0.05)) if c.sum() >= 10]
    for c in components(mask_teal(rgb) & on & (P[..., 2] < 0.05)):
        if c.sum() < 15: continue
        pts = P[c]
        ext = np.ptp(pts[:, :2], 0).max()
        if ext > 0.12: continue
        hc = np.median(pts, 0)
        # nearest shaft
        bs, bd = None, 1e9
        for s in shafts:
            sp = P[s]
            dmin = np.min(np.linalg.norm(sp[:, :2] - hc[:2], axis=1))
            if dmin < bd: bd, bs = dmin, s
        if bs is None or bd > 0.08: continue
        cands.append((bs.sum(), c, bs, hc))
    out["tools"] = []
    cands.sort(key=lambda t: -t[0])
    for _, c, s, hc in cands:
        sp = P[s][:, :2]
        hp = P[c][:, :2]
        allp = np.vstack([hp, sp])
        mu = allp.mean(0)
        u, sv, vt = np.linalg.svd(allp - mu)
        a = vt[0]
        if np.dot(sp.mean(0) - hc[:2], a) < 0: a = -a
        proj = (sp - hc[:2]) @ a
        out["tools"].append({"handle": hc, "handle_top": float(np.percentile(P[c][:, 2], 90)),
                             "tool_axis": a, "tool_len": float(proj.max()), "shaft_px": int(s.sum())})
    if cands:
        _, c, s, hc = cands[0]
        sp = P[s][:, :2]
        hp = P[c][:, :2]
        allp = np.vstack([hp, sp])
        mu = allp.mean(0)
        u, sv, vt = np.linalg.svd(allp - mu)
        a = vt[0]
        if np.dot(sp.mean(0) - hc[:2], a) < 0: a = -a
        proj = (sp - hc[:2]) @ a
        out["handle"] = hc
        out["handle_top"] = float(np.percentile(P[c][:, 2], 90))
        out["tool_axis"] = a
        out["tool_len"] = float(proj.max())
        out["shaft_px"] = int(s.sum())
    return out
