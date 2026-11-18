import json, os
import numpy as np
from geom import *

WS = "/workspace"
COLORS = {"red": lambda h: (h < 12) | (h > 340), "orange": lambda h: (h >= 15) & (h < 50),
          "yellow": lambda h: (h >= 50) & (h < 75), "green": lambda h: (h >= 85) & (h < 165),
          "cyan": lambda h: (h >= 165) & (h < 200),
          "blue": lambda h: (h >= 205) & (h < 258), "purple": lambda h: (h >= 262) & (h < 330)}

def frame_files(fr):
    return {(f["camera"], f["kind"]): os.path.join(WS, f["path"]) for f in fr["files"]}

def cam_world(fr_or_obs, cam, rgb_path, depth_path):
    rgb = load_rgb(rgb_path); dep = load_depth(depth_path)
    Wm = world_map(dep, fr_or_obs["camera_intrinsics"][cam]["matrix"], fr_or_obs["camera_poses"][cam])
    return rgb, Wm

def detect_cubes_in_frame(fr, min_px=12):
    """Return list of (color, xyz) of saturated cube blobs in both cameras."""
    files = frame_files(fr); out = []
    for cam in ("front", "wrist"):
        if (cam, "rgb") not in files or (cam, "depth") not in files: continue
        try:
            rgb, Wm = cam_world(fr, cam, files[(cam, "rgb")], files[(cam, "depth")])
        except Exception:
            continue
        h, s, v = hsv(rgb)
        base = (s > 0.8) & (v > 0.12) & np.isfinite(Wm[..., 2]) & (Wm[..., 2] > 0.003) & (Wm[..., 2] < 0.06) \
            & (np.abs(Wm[..., 0]) < 0.45) & (np.abs(Wm[..., 1]) < 0.45)
        for name, fn in COLORS.items():
            m = base & fn(h)
            if m.sum() < min_px: continue
            P = Wm[m]
            for c in cluster_xy(P, 0.03, min_px):
                npx = int((np.linalg.norm(P[:, :2] - c[:2], axis=1) < 0.03).sum())
                out.append((name, cam, c, npx))
    return out

def cluster_xy(P, rad, min_n):
    pts = P[:, :2]; lab = -np.ones(len(pts), int); cents = []
    order = np.arange(len(pts)); c = 0
    for i in order:
        if lab[i] >= 0: continue
        cen = pts[i]
        for _ in range(5):
            sel = np.linalg.norm(pts - cen, axis=1) < rad
            cen = pts[sel].mean(0)
        sel = (np.linalg.norm(pts - cen, axis=1) < rad) & (lab < 0)
        lab[sel] = c; c += 1
        if sel.sum() >= min_n: cents.append(np.array([cen[0], cen[1], float(np.median(P[sel, 2]))]))
    return cents

def video_cube_locations(manifest_path, merge=0.03):
    """Location-first tracking: each physical cube location gets one colour by pixel majority."""
    m = json.load(open(manifest_path))
    locs = []
    for fr in m["frames"]:
        if fr.get("phase") != "video_demo": continue
        for color, cam, c, npx in detect_cubes_in_frame(fr):
            best = None
            for L in locs:
                d = np.linalg.norm(L["xy"] - c[:2])
                if d < merge and (best is None or d < best[0]): best = (d, L)
            if best is None:
                L = {"sum": np.zeros(2), "w": 0.0, "xy": c[:2].copy(), "votes": {}, "frames": set(), "cams": set()}
                locs.append(L)
            else:
                L = best[1]
            L["sum"] += c[:2] * npx; L["w"] += npx; L["xy"] = L["sum"] / L["w"]
            L["votes"][color] = L["votes"].get(color, 0) + npx
            L["frames"].add(fr.get("frame_index")); L["cams"].add(cam)
    out = []
    for L in locs:
        color = max(L["votes"], key=L["votes"].get)
        out.append({"color": color, "xy": L["xy"], "n": len(L["frames"]), "votes": L["votes"], "cams": L["cams"]})
    return out

def video_cube_tracks(manifest_path):
    """Accumulate cube detections over the conditioning video -> list of dict(color, xy, n)."""
    m = json.load(open(manifest_path))
    tracks = []
    for fr in m["frames"]:
        if fr.get("phase") != "video_demo": continue
        for color, cam, c, npx in detect_cubes_in_frame(fr):
            for t in tracks:
                if t["color"] == color and np.linalg.norm(t["xy"] - c[:2]) < 0.035:
                    t["n"] += 1; t["sum"] += c[:2]; t["xy"] = t["sum"] / t["n"]; t["cams"].add(cam); break
            else:
                tracks.append({"color": color, "xy": c[:2].copy(), "sum": c[:2].copy(), "n": 1, "cams": {cam}})
    return tracks

def container_tops(Wm, center=None, radius=0.07, zlo=0.055, zhi=0.095, rgb=None):
    """Find container top faces near center from a world map; returns list of (xy, yaw, npts)."""
    m = np.isfinite(Wm[..., 2]) & (Wm[..., 2] > zlo) & (Wm[..., 2] < zhi)
    if rgb is not None:
        h, s, v = hsv(rgb); m &= (s < 0.25) & (v > 0.45)
    P = Wm[m]
    if center is not None and len(P):
        P = P[np.linalg.norm(P[:, :2] - np.asarray(center)[:2], axis=1) < radius]
    res = []
    if len(P) < 10: return res
    for c in cluster_xy(P, 0.028, 15):
        Q = P[np.linalg.norm(P[:, :2] - c[:2], axis=1) < 0.03][:, :2]
        cen = (Q.max(0) + Q.min(0)) / 2
        best = None
        for a in np.arange(0, 90, 2.0):
            t = np.deg2rad(a); R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            q = (Q - Q.mean(0)) @ R.T; lo, hi = q.min(0), q.max(0); A = np.prod(hi - lo)
            if best is None or A < best[0]: best = (A, a, (lo + hi) / 2, hi - lo)
        A, a, mid, ext = best
        t = np.deg2rad(a); R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        cen = Q.mean(0) + R.T @ mid
        yaw = ((a + 45) % 90) - 45
        res.append({"xy": cen, "yaw_deg": yaw, "n": len(Q), "ext": ext})
    return res

def obs_images(reply):
    """Decode rgb/depth attachments of an MCP reply -> (meta, {(cam,kind): array})."""
    import base64, io
    from PIL import Image
    content = reply.get("content", []) if isinstance(reply, dict) else reply
    meta = json.loads(next(b["text"] for b in content if b.get("type") == "text"))
    out = {}
    for a in meta.get("attachments", []):
        b = content[a["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.asarray(im)
        if a["kind"] == "depth":
            out[(a["camera"], "depth")] = arr.astype(np.float32) * meta["depth"][a["camera"]].get("meters_per_unit", 0.001)
        else:
            out[(a["camera"], "rgb")] = np.asarray(im.convert("RGB")).astype(np.float32)
    return meta, out

def obs_world(reply, cam):
    meta, ims = obs_images(reply)
    Wm = world_map(ims[(cam, "depth")], meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam])
    return meta, ims[(cam, "rgb")], Wm
