"""Classify the manner shown in the conditioning video: 'pick', 'push' or 'tool'."""
import json
import math
from pathlib import Path

import numpy as np

import mcio
import perception as P
from perception import Cloud

KEYWORDS = {
    "pick": ["pick up", "pick it", "pick the", "lift the", "grasp the cube", "carry"],
    "push": ["push"],
    "tool": ["peg", "stick", "tool", "hook", "rod", "pull", "drag"],
}


def instruction_prior(text):
    t = (text or "").lower()
    hits = {k: any(w in t for w in ws) for k, ws in KEYWORDS.items()}
    n = sum(hits.values())
    if n == 1:
        return [k for k, v in hits.items() if v][0]
    return None


def load_video_frames(man):
    frames = sorted(man.get("frames", []), key=lambda f: f.get("frame_index", 0))
    vid = [f for f in frames if f.get("phase") == "video_demo"]
    exe = [f for f in frames if f.get("phase") != "video_demo"]
    return vid, exe


def cloud_from_frame(frame, perceiver, K, pose, variant, stride=2):
    imgs = mcio.frame_images(frame)
    f = imgs.get("front", {})
    d, rgb = f.get("depth"), f.get("rgb")
    if d is None or K is None or pose is None or variant in (None, "tool"):
        return None
    H, W = d.shape
    import geom
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    dd = d[vv, uu]
    xyz = geom.backproject(uu.ravel(), vv.ravel(), dd.ravel(), K, pose, variant).reshape(vv.shape + (3,))
    col = rgb[vv, uu].astype(float) if rgb is not None else np.full(vv.shape + (3,), 128.0)
    return Cloud(xyz, dd > 0, col, np.stack([uu, vv], -1), "front")


def state_features(vid, table_z):
    S = [f.get("state") for f in vid]
    S = np.array([s for s in S if isinstance(s, list) and len(s) == 8], float)
    out = {"n_states": int(len(S))}
    if len(S) < 5:
        out["informative"] = False
        return out
    pos = S[:, :3]
    g = S[:, 6] + S[:, 7]
    travel = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    out.update(travel=travel, g_max=float(g.max()), g_min=float(g.min()))
    out["informative"] = travel > 0.05 or (g.max() - g.min()) > 0.01
    gopen = g.max()
    closed = g < gopen - 0.012
    out["closed_frac"] = float(closed.mean())
    # held segments: closed and gripper settled (two consecutive similar widths)
    segs = []
    i = 0
    while i < len(g):
        if closed[i]:
            j = i
            while j + 1 < len(g) and closed[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    out["segments"] = []
    best = None
    for a, b in segs:
        w = g[a:b + 1]
        settled = float(np.median(w[len(w) // 3:])) if len(w) >= 3 else float(w[-1])
        z = pos[a:b + 1, 2]
        seg = {"a": a, "b": b, "width": settled, "z_grasp": float(z[0]), "z_max": float(z.max()),
               "lift": float(z.max() - z[: max(1, len(z) // 4)].min()),
               "xy_grasp": pos[a, :2].tolist(), "travel": float(np.sum(np.linalg.norm(np.diff(pos[a:b + 1], axis=0), axis=1))) if b > a else 0.0,
               "released": bool(b < len(g) - 1), "R_grasp": S[a, 3:6].tolist()}
        out["segments"].append(seg)
        if best is None or (seg["b"] - seg["a"]) > (best["b"] - best["a"]):
            best = seg
    out["main"] = best
    if table_z is not None:
        out["z_min_rel"] = float(pos[:, 2].min() - table_z)
    return out


def image_features(vid, perceiver, K, pose, variant, table_z, log=print, max_frames=36):
    out = {}
    if not vid:
        return out
    idx = np.unique(np.linspace(0, len(vid) - 1, min(max_frames, len(vid))).astype(int))
    c0 = cloud_from_frame(vid[0], perceiver, K, pose, variant)
    if c0 is None:
        out["ok"] = False
        return out
    tz = table_z if table_z is not None else perceiver.fit_table(c0)
    comps, _ = perceiver.components(c0, tz)
    cubes, tools, others = P.classify_objects(comps)
    out["n_cubes"], out["n_tools"] = len(cubes), len(tools)
    if not cubes:
        out["ok"] = False
        return out
    cube = cubes[0]
    out["cube0"] = cube["center"].tolist()
    out["cube_color"] = cube["color"].tolist()
    out["cube_h"] = cube["hmax"]
    tool = tools[0] if tools else None
    if tool is not None:
        out["tool0"] = tool["center"].tolist()
        out["tool_color"] = tool["color"].tolist()
        out["tool_axis"] = tool["axis"].tolist()
        out["tool_pts"] = tool["pts"][:, :2]
    cube_track, tool_track = [], []
    st = [f.get("state") for f in vid]
    tool_xy0 = tool["pts"][:, :2] if tool is not None else None
    occ0 = None
    for k in idx:
        cl = c0 if k == 0 else cloud_from_frame(vid[k], perceiver, K, pose, variant)
        if cl is None:
            continue
        s = st[k] if isinstance(st[k], list) and len(st[k]) == 8 else None
        ex = dict(exclude_xy=s[:2], exclude_r=0.05, exclude_zmin=s[2] - 0.01) if s else {}
        ct = P.color_track(cl, tz, cube["color"], near=None, thr=0.10, **ex)
        if ct is not None and ct["n"] >= 6:
            cube_track.append((int(k), ct["center"].tolist(), float(ct["zmax"] - tz), ct["n"]))
        if tool is not None:
            occ = tool_occupancy(cl, tz, tool, tool_xy0)
            if occ0 is None:
                occ0 = max(occ, 1)
            over = False
            if s is not None:
                over = float(np.min(np.linalg.norm(tool_xy0 - np.array(s[:2]), axis=1))) < 0.08
            tool_track.append((int(k), occ / occ0, bool(over)))
    out["cube_track"] = cube_track
    out["tool_track"] = tool_track
    if cube_track:
        hs = np.array([c[2] for c in cube_track])
        out["cube_lift"] = float(hs.max() - cube["hmax"])
        out["cube_disp"] = float(np.linalg.norm(np.array(cube_track[-1][1]) - cube["center"]))
        out["cube_final"] = cube_track[-1][1]
    if tool_track:
        vis = [t[1] for t in tool_track if not t[2]]
        tail = vis[-3:] if vis else []
        out["tool_left_frac"] = float(1 - max(tail)) if tail else 0.0
    out["ok"] = True
    return out


def tool_occupancy(cl, tz, tool, xy0):
    """How many tool-coloured low points still lie on the tool's initial footprint."""
    h = cl.xyz[..., 2] - tz
    m = cl.valid & (h > 0.003) & (h < tool["hmax"] + 0.01)
    if m.sum() == 0:
        return 0
    pts = cl.xyz[m][:, :2]
    d = P.color_dist(cl.rgb[m], tool["color"]) < 0.10
    pts = pts[d]
    if len(pts) == 0:
        return 0
    lo, hi = xy0.min(0) - 0.01, xy0.max(0) + 0.01
    inb = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[inb]
    if len(pts) == 0:
        return 0
    cnt = 0
    for i in range(0, len(pts), 512):
        dd = np.linalg.norm(pts[i:i + 512, None, :] - xy0[None, ::3, :], axis=2).min(1)
        cnt += int((dd < 0.008).sum())
    return cnt


def decide(sf, imf, prior, log=print):
    votes = {"pick": 0.0, "push": 0.0, "tool": 0.0}
    why = []
    if sf.get("informative"):
        main = sf.get("main")
        held = [s for s in sf.get("segments", []) if s["width"] > 0.006 and (s["b"] - s["a"]) >= 2]
        if not held:
            votes["push"] += 3; why.append("state: no grasp with object width")
        else:
            seg = max(held, key=lambda s: s["b"] - s["a"])
            gxy = np.array(seg["xy_grasp"])
            dc = dt = None
            if imf.get("cube0") is not None:
                dc = float(np.linalg.norm(gxy - np.array(imf["cube0"])))
            if imf.get("tool_pts") is not None:
                dt = float(np.min(np.linalg.norm(imf["tool_pts"] - gxy, axis=1)))
            why.append(f"state: held width {seg['width']:.3f} lift {seg['lift']:.3f} dcube {dc} dtool {dt}")
            if dc is not None and dt is not None:
                if dc < 0.035 and dc < dt:
                    votes["pick"] += 3
                elif dt < 0.03 and dt < dc:
                    votes["tool"] += 3
            elif dc is not None and dc < 0.035:
                votes["pick"] += 2
            elif dt is not None and dt < 0.03:
                votes["tool"] += 2
            if not seg["released"]:
                votes["tool"] += 0.5
            else:
                votes["pick"] += 0.5
            if abs(seg["width"] - (imf.get("cube_h") or 0.04)) < 0.008:
                votes["pick"] += 0.7
            else:
                votes["tool"] += 0.7
    if imf.get("ok"):
        if imf.get("cube_lift", 0) > 0.035:
            votes["pick"] += 2; why.append(f"img: cube lifted {imf['cube_lift']:.3f}")
        if imf.get("tool_left_frac", 0) > 0.6:
            votes["tool"] += 2; why.append(f"img: tool left its place {imf.get('tool_left_frac', 0):.2f}")
        if imf.get("cube_disp", 0) > 0.03 and imf.get("cube_lift", 0) < 0.02 and imf.get("tool_left_frac", 0) < 0.3:
            votes["push"] += 1.5; why.append("img: cube slid, tool static")
    if prior:
        votes[prior] += 2.5; why.append(f"instruction prior {prior}")
    best = max(votes, key=lambda k: votes[k])
    if votes[best] == 0:
        best = "pick"
        why.append("no evidence: default pick")
    log("manner votes", votes, "->", best, "|", "; ".join(why))
    return best, votes, why
