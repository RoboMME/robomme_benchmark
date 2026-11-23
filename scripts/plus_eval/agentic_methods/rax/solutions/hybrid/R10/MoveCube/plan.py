"""Scene analysis: video grasp detection, tool matching, cube/target localisation."""
import json, math
import numpy as np
from percep import *

def width_profile(b, origin_pt, nb=8):
    """perpendicular width profile along blob axis, oriented starting at end nearest origin_pt"""
    xy = b["pts"][:, :2]; c = b["center"]; ax = b["axis"]
    if np.dot(origin_pt[:2] - c, ax) > 0:
        ax = -ax
    # now ax points away from origin end
    s = (xy - c) @ ax; perp = (xy - c) @ np.array([-ax[1], ax[0]])
    lo, hi = s.min(), s.max()
    prof = []
    for k in range(nb):
        a = lo + (hi - lo) * k / nb; bb = lo + (hi - lo) * (k + 1) / nb
        sel = (s >= a) & (s <= bb)
        prof.append(float(perp[sel].max() - perp[sel].min()) if sel.sum() > 1 else 0.0)
    return np.array(prof), ax

def is_robot(b):
    return b["center"][0] < -0.48 and abs(b["center"][1]) < 0.15 and b["zmax"] > 0.06

def classify_static(blobs):
    cube = target = None
    for b in blobs:
        h = dict(zip(CLASSES, b["hist"]))
        if h["red"] > 0.6 and b["length"] < 0.09:
            if cube is None or b["n"] > cube["n"]:
                cube = b
        if h["purple"] > 0.35 and b["zmax"] < 0.03:
            if target is None or b["n"] > target["n"]:
                target = b
    return cube, target

def analyze_video(frames, log=print):
    vid = [f for f in frames if f["phase"] == "video_demo"]
    if not vid:
        return None
    S = np.array([f["state"] for f in vid])
    g = S[:, 6] + S[:, 7]
    closed = np.where(g < 0.075)[0]
    info = dict(n=len(vid))
    if len(closed) == 0:
        info["manner"] = "no_grasp"
        return info
    i0 = int(closed[0])
    # wait until fingers settle
    i1 = i0
    while i1 + 1 < len(vid) and abs(g[i1 + 1] - g[i1]) > 1e-4 and i1 < i0 + 20:
        i1 += 1
    info["grasp_index"] = i1; info["grasp_state"] = S[i1].tolist(); info["grip_width"] = float(g[i1])
    gp = S[i1, :3]
    f0 = vid[0]
    rgb, dep = load_frame(f0)
    blobs, lab, pw = scene_blobs(rgb, dep, f0["camera_intrinsics"]["front"], f0["camera_poses"]["front"])
    blobs = [b for b in blobs if not is_robot(b)]
    # nearest blob to grasp point (xy)
    best = None; bd = 1e9
    for b in blobs:
        d = np.min(np.linalg.norm(b["pts"][:, :2] - gp[:2], axis=1))
        if d < bd:
            bd = d; best = b
    info["grasp_blob_dist"] = float(bd)
    if best is None or bd > 0.05:
        info["manner"] = "unknown"; return info
    # colour near grasp point
    cc = color_class(rgb)
    near = best["mask"] & (np.linalg.norm(pw[..., :2] - gp[:2], axis=-1) < 0.025)
    labs = cc[near]
    vals, cnt = np.unique(labs, return_counts=True) if len(labs) else (np.array(["other"]), np.array([1]))
    gcol = str(vals[np.argmax(cnt)])
    prof, ax = width_profile(best, gp)
    h = dict(zip(CLASSES, best["hist"]))
    info.update(grasp_color=gcol, blob=describe(best), profile=prof.round(3).tolist())
    if h["red"] > 0.5:
        info["manner"] = "cube_pick"
    elif h["yellow"] > 0.5 and best["length"] < 0.08:
        info["manner"] = "ball"
    else:
        info["manner"] = "tool"
    info["_blob"] = best; info["_gp"] = gp
    return info

def tool_feature(b, gcol):
    cc_hist = b["hist"]
    return cc_hist, b["length"], b["zmax"]

def match_tool(vinfo, blobs, cube, target):
    """pick exec blob most similar to the video-grasped object"""
    vb = vinfo["_blob"]; gcol = vinfo["grasp_color"]
    cands = []
    for b in blobs:
        if is_robot(b) or b is cube or b is target:
            continue
        h = dict(zip(CLASSES, b["hist"]))
        if h["blue"] > 0.5 and b["n"] > 1500:
            continue  # ramp
        if b["n"] < 25:
            continue
        dh = np.abs(b["hist"] - vb["hist"]).sum()
        dl = abs(b["length"] - vb["length"]) / 0.1
        dz = abs(b["zmax"] - vb["zmax"]) / 0.02
        # must contain grasp colour
        ci = CLASSES.index(gcol) if gcol in CLASSES else 7
        pen = 0 if b["hist"][ci] > 0.05 else 2.0
        cands.append((dh + 0.5 * dl + 0.3 * dz + pen, b))
    cands.sort(key=lambda x: x[0])
    return cands

def grasp_pose_for(b, gcol, rgb, pw):
    cc = color_class(rgb)
    m = b["mask"] & (cc == gcol)
    if m.sum() < 4:
        m = b["mask"]
    pts = pw[m]
    gxy = np.median(pts[:, :2], axis=0)
    ztop = float(np.percentile(pts[:, 2], 90))
    # handle axis: from colour region if elongated, else blob axis
    ax = b["axis"]
    return gxy, ztop, ax

def color_center(rgb, pw, cls, zmin=-0.01, zmax=0.08):
    cc = color_class(rgb)
    z = pw[..., 2]
    m = (cc == cls) & np.isfinite(z) & (z > zmin) & (z < zmax)
    if m.sum() < 5:
        return None
    lab, n = label(m)
    sizes = [(lab == i).sum() for i in range(1, n + 1)]
    k = int(np.argmax(sizes)) + 1
    mm = lab == k
    if mm.sum() < 5:
        return None
    return np.median(pw[mm][:, :2], axis=0)
