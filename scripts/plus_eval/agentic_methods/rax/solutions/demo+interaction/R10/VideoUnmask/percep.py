import json, math
from pathlib import Path
import numpy as np
from PIL import Image

WS = Path("/workspace")

def quat_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def backproject(u, v, d, K, pose):
    fx, cx = K[0][0], K[0][2]; fy, cy = K[1][1], K[1][2]
    cam = np.stack([(u-cx)*d/fx, -(v-cy)*d/fy, -d], axis=-1)
    R = quat_rot(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    return cam @ R.T + t

def hsv(rgb):
    a = rgb.astype(np.float32)/255.0
    r, g, b = a[...,0], a[...,1], a[...,2]
    mx = a.max(-1); mn = a.min(-1); df = mx-mn+1e-6
    h = np.where(mx==r, ((g-b)/df) % 6, np.where(mx==g, (b-r)/df+2, (r-g)/df+4))*60
    s = df/(mx+1e-6)
    return h, s, mx

def color_masks(rgb):
    h, s, v = hsv(rgb)
    sat = (s > 0.75) & (v > 0.15)
    return {
        "red": sat & ((h < 8) | (h > 340)),
        "green": sat & (h > 90) & (h < 150),
        "blue": sat & (h > 210) & (h < 255),
    }

def blobs(mask, min_px=4):
    """simple connected components; returns list of (pixel list arrays)"""
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, bool); out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]: continue
        q = deque([(y0, x0)]); seen[y0, x0] = True; pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(pts) >= min_px: out.append(np.array(pts))
    return out

def load_frame(fr, cam):
    files = {(f["camera"], f["kind"]): f["path"] for f in fr["files"]}
    rgb = np.array(Image.open(WS/files[(cam, "rgb")]).convert("RGB"))
    dp = files.get((cam, "depth"))
    depth = np.array(Image.open(WS/dp)).astype(np.float32)*0.001 if dp else None
    return rgb, depth

def detect_cubes_frame(fr, cams=("front", "wrist")):
    dets = []
    for cam in cams:
        rgb, depth = load_frame(fr, cam)
        if depth is None: continue
        K = fr["camera_intrinsics"][cam]["matrix"]; pose = fr["camera_poses"][cam]
        for col, m in color_masks(rgb).items():
            for pts in blobs(m):
                vv, uu = pts[:,0], pts[:,1]
                d = depth[vv, uu]; ok = d > 0
                if ok.sum() < 3: continue
                P = backproject(uu[ok].astype(float), vv[ok].astype(float), d[ok], K, pose)
                c = np.median(P, axis=0)
                dets.append({"cam": cam, "color": col, "n": int(len(pts)), "xyz": c.tolist(),
                             "uv": [float(uu.mean()), float(vv.mean())]})
    return dets

import base64, io

def parse_reply(result):
    blocks = result.get("content", [])
    meta = json.loads(blocks[0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = blocks[a["content_index"]]
        data = base64.b64decode(b["data"])
        im = Image.open(io.BytesIO(data))
        if a["kind"] == "rgb":
            imgs[(a["camera"], "rgb")] = np.array(im.convert("RGB"))
        else:
            imgs[(a["camera"], "depth")] = np.array(im).astype(np.float32)*0.001
    return meta, imgs

def white_mask(rgb):
    h, s, v = hsv(rgb)
    return (s < 0.18) & (v > 0.55)

def detect_containers(rgb, depth, K, pose, zmin=0.015, min_px=20):
    """returns list of containers: xy centre (of top/visible points), top z, yaw (rad, mod pi/2), n"""
    m = white_mask(rgb) & (depth > 0)
    H, W = m.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu.astype(float), vv.astype(float), depth, K, pose)
    m &= (P[...,2] > zmin) & (P[...,2] < 0.25)
    out = []
    for pts in blobs(m, min_px):
        Q = P[pts[:,0], pts[:,1]]
        ztop = np.percentile(Q[:,2], 90)
        top = Q[Q[:,2] > ztop - 0.012]
        if len(top) < 5: top = Q
        xy = top[:, :2]
        c = (xy.max(0) + xy.min(0))/2
        # yaw via minimal-area bounding box over angle search
        best = None
        for a in np.linspace(0, math.pi/2, 46, endpoint=False):
            R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            p = (xy - xy.mean(0)) @ R.T
            ext = p.max(0) - p.min(0)
            area = ext[0]*ext[1]
            if best is None or area < best[0]:
                cen = (p.max(0)+p.min(0))/2
                best = (area, a, ext, R.T @ cen + xy.mean(0))
        out.append({"xy": best[3].tolist(), "ztop": float(ztop), "yaw": float(best[1]),
                    "ext": best[2].tolist(), "n": int(len(pts)), "uv": [float(pts[:,1].mean()), float(pts[:,0].mean())]})
    return out

def detect_containers_geom(rgb, depth, K, pose, zlo=0.05, zhi=0.10, min_px=20):
    """appearance-independent: blobs of points whose height lies in the container-top band"""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = backproject(uu.astype(float), vv.astype(float), depth, K, pose)
    m = (depth > 0) & (P[...,2] > zlo) & (P[...,2] < zhi)
    out = []
    for pts in blobs(m, min_px):
        Q = P[pts[:,0], pts[:,1]]
        ztop = np.percentile(Q[:,2], 90)
        top = Q[Q[:,2] > ztop - 0.010]
        if len(top) < 5: top = Q
        xy = top[:, :2]
        best = None
        for a in np.linspace(0, math.pi/2, 46, endpoint=False):
            R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
            p = (xy - xy.mean(0)) @ R.T
            ext = p.max(0) - p.min(0)
            area = ext[0]*ext[1]
            if best is None or area < best[0]:
                cen = (p.max(0)+p.min(0))/2
                best = (area, a, ext, R.T @ cen + xy.mean(0))
        out.append({"xy": best[3].tolist(), "ztop": float(ztop), "yaw": float(best[1]),
                    "ext": best[2].tolist(), "n": int(len(pts)), "uv": [float(pts[:,1].mean()), float(pts[:,0].mean())]})
    return out
