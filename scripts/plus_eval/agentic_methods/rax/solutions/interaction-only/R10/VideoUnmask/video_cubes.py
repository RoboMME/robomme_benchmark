"""Cube detections from all conditioning frames (front + wrist), clustered in world xy."""
import json
import numpy as np
from PIL import Image
from perception import WS, backproject, components, color_mask, frame_files


def black_mask(hsv):
    return (hsv[..., 2] < 45)


def detect_frame(fr, colors, cams=("front", "wrist")):
    files = frame_files(fr)
    dets = []
    for cam in cams:
        if (cam, "rgb") not in files or (cam, "depth") not in files:
            continue
        rgb = Image.open(WS / files[(cam, "rgb")]).convert("RGB")
        hsv = np.asarray(rgb.convert("HSV")).astype(int)
        depth = np.asarray(Image.open(WS / files[(cam, "depth")])).astype(float) * 0.001
        K = fr["camera_intrinsics"][cam]["matrix"]; pose = fr["camera_poses"][cam]
        for c in colors:
            m = black_mask(hsv) if c == "black" else color_mask(hsv, c)
            m &= depth > 0
            for ys, xs in components(m, 10 if cam == "front" else 25):
                P = backproject(xs, ys, depth[ys, xs], K, pose)
                xyz = np.median(P, 0)
                # cube on table: centre-top below 0.06, above table; reject robot parts
                zmin = 0.008 if c == "black" else 0.012
                if not (zmin < xyz[2] < 0.06) or xyz[0] < -0.40 or xyz[0] > 0.25:
                    continue
                ext = P[:, :2].max(0) - P[:, :2].min(0)
                if ext.max() > 0.08:
                    continue
                dets.append({"color": c, "cam": cam, "xyz": xyz, "n": len(xs), "frame": fr["frame_index"]})
    return dets


def cube_clusters(manifest_path, colors, log=print, stride=2, radius=0.025):
    man = json.loads((WS / manifest_path).read_text())
    frames = sorted([f for f in man["frames"] if f["phase"] == "video_demo"], key=lambda f: f["frame_index"])
    sel = frames[::stride]
    if frames and frames[-1] not in sel:
        sel.append(frames[-1])
    clusters = []  # each: color, list of xyz, last frame
    for fr in sel:
        for d in detect_frame(fr, colors):
            best, bd = None, radius
            for cl in clusters:
                if cl["color"] != d["color"]:
                    continue
                dist = np.linalg.norm(cl["last_xyz"][:2] - d["xyz"][:2])
                if dist < bd:
                    best, bd = cl, dist
            if best is None:
                clusters.append({"color": d["color"], "pts": [d["xyz"]], "last_xyz": d["xyz"], "count": 1, "pix": d["n"],
                                 "first": d["frame"], "last": d["frame"], "cams": {d["cam"]}})
            else:
                best["pts"].append(d["xyz"]); best["count"] += 1; best["last"] = d["frame"]; best["pix"] += d["n"]
                best["cams"].add(d["cam"])
                # weight wrist (closer) detections; use running recent median
                best["last_xyz"] = np.median(np.array(best["pts"][-6:]), 0)
    out = [c for c in clusters if c["count"] >= 2]
    for c in out:
        c["xy"] = c["last_xyz"][:2].tolist()
    # same-colour merge (split blobs), then colour conflicts at one location: keep majority pixels
    merged = []
    for c in sorted(out, key=lambda c: -c["pix"]):
        for o in merged:
            if o["color"] == c["color"] and np.linalg.norm(np.array(o["xy"]) - np.array(c["xy"])) < 0.035:
                o["pix"] += c["pix"]; o["count"] += c["count"]; o["cams"] |= c["cams"]
                break
        else:
            merged.append(c)
    keep = []
    for c in sorted(merged, key=lambda c: -c["pix"]):
        clash = [o for o in keep if np.linalg.norm(np.array(o["xy"]) - np.array(c["xy"])) < 0.03]
        if clash:
            log(f"drop {c['color']} cluster at ({c['xy'][0]:.3f},{c['xy'][1]:.3f}) pix={c['pix']}: overlaps {clash[0]['color']} pix={clash[0]['pix']}")
            continue
        keep.append(c)
    out = keep
    for c in out:
        log(f"cube cluster {c['color']}: xy=({c['xy'][0]:.3f},{c['xy'][1]:.3f}) z={c['last_xyz'][2]:.3f} n={c['count']} pix={c['pix']} frames {c['first']}-{c['last']} cams={sorted(c['cams'])}")
    return out
