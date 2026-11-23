"""Perception from one robomme_observe reply: decode RGB-D, back-project, segment."""
import base64, io, json
import numpy as np
from PIL import Image


def decode(reply):
    meta = json.loads(reply["content"][0]["text"])
    imgs = {}
    for a in meta.get("attachments", []):
        b = reply["content"][a["content_index"]]
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if a["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        else:
            arr = arr.astype(np.float64) * meta["depth"][a["camera"]].get("meters_per_unit", 0.001)
        imgs[(a["camera"], a["kind"])] = arr
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def backproject(meta, depth, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"])
    t = np.array(pose["position_m"])
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = depth
    xc = (u - K[0, 2]) * d / K[0, 0]
    yc = -(v - K[1, 2]) * d / K[1, 1]
    zc = -d
    P = np.stack([xc, yc, zc], -1) @ R.T + t
    P[d <= 0] = np.nan
    return P


def components(mask, min_size=4):
    """4-connected components; returns list of (N,2) arrays of (v,u)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        pts = []
        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        if len(pts) >= min_size:
            out.append(np.array(pts))
    return out


def color_mask(rgb, color):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if color == "green":
        return (G > 90) & (G - R > 40) & (G - B > 35)
    if color == "blue":
        return (B > 90) & (B - R > 50) & (B - G > 25)
    if color == "yellow":
        return (R > 140) & (G > 140) & (B < 100) & (abs(R - G) < 60)
    if color in ("purple", "magenta", "pink"):
        return (R > 90) & (B > 90) & (G < 0.7 * np.minimum(R, B))
    if color == "orange":
        return (R > 180) & (G > 70) & (G < 150) & (B < 60)
    # red default
    return (R > 120) & (G < 70) & (B < 70) & (R - G > 90)


def analyse(meta, imgs, cam="front", color="red"):
    rgb = imgs[(cam, "rgb")].astype(np.int32)
    P = backproject(meta, imgs[(cam, "depth")], cam)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    z = P[..., 2]
    valid = np.isfinite(z)
    zz = np.where(valid, z, 9.0)
    near_table = valid & (zz > -0.02) & (zz < 0.10) & (P[..., 0] > -0.45) & (P[..., 0] < 0.25)
    red = near_table & (zz > 0.012) & color_mask(rgb, color)
    mx = np.maximum(np.maximum(R, G), B)
    mn = np.minimum(np.minimum(R, G), B)
    grey = near_table & (mx - mn < 40) & (mx > 90)
    res = {"cam": cam}
    comps = components(red, 3)
    if comps:
        c = max(comps, key=len)
        pts = P[c[:, 0], c[:, 1]]
        res["red"] = {"n": len(c), "uv": [float(c[:, 1].mean()), float(c[:, 0].mean())],
                      "xyz": np.nanmedian(pts, 0).tolist(), "zmax": float(np.nanmax(pts[:, 2]))}
    objs = []
    for c in components(grey, 6):
        pts = P[c[:, 0], c[:, 1]]
        objs.append({"n": len(c), "uv": [float(c[:, 1].mean()), float(c[:, 0].mean())],
                     "xyz": np.nanmedian(pts, 0).tolist(),
                     "zmax": float(np.nanpercentile(pts[:, 2], 90)),
                     "ext": (np.nanmax(pts[:, :2], 0) - np.nanmin(pts[:, :2], 0)).tolist()})
    res["grey"] = objs
    return res, P


def find_discs(meta, imgs, cam="front"):
    """Flat target discs: pixels whose world height is in the disc-top band (colour independent)."""
    P = backproject(meta, imgs[(cam, "depth")], cam)
    z = np.nan_to_num(P[..., 2], nan=9.0)
    x = np.nan_to_num(P[..., 0], nan=9.0)
    y = np.nan_to_num(P[..., 1], nan=9.0)
    m = (z > 0.0045) & (z < 0.0135) & (x > -0.5) & (x < 0.3) & (np.abs(y) < 0.45)
    out = []
    for c in components(m, 15):
        pts = P[c[:, 0], c[:, 1]]
        ext = (np.nanmax(pts[:, :2], 0) - np.nanmin(pts[:, :2], 0))
        if ext.max() > 0.12:
            continue
        col = imgs[(cam, "rgb")][c[:, 0], c[:, 1]].astype(float)
        R, G, B = col[:, 0], col[:, 1], col[:, 2]
        sat = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)
        redfrac = float(np.mean(sat > 70))  # highlighted (non-grey) fraction
        out.append({"n": len(c), "xyz": np.nanmedian(pts, 0).tolist(), "ext": ext.tolist(),
                    "ctr": ((np.nanmax(pts[:, :2], 0) + np.nanmin(pts[:, :2], 0)) / 2).tolist(),
                    "redfrac": redfrac})
    return out


def find_buttons(meta, imgs, cam="front"):
    rgb = imgs[(cam, "rgb")].astype(np.int32)
    P = backproject(meta, imgs[(cam, "depth")], cam)
    z = np.nan_to_num(P[..., 2], nan=9.0)
    x = np.nan_to_num(P[..., 0], nan=9.0)
    mx = rgb.max(-1); mn = rgb.min(-1)
    grey = (mx - mn < 40) & (mx > 60)
    m = grey & (z > 0.02) & (z < 0.05) & (x > -0.5) & (x < 0.3)
    out = []
    for c in components(m, 8):
        pts = P[c[:, 0], c[:, 1]]
        ext = (np.nanmax(pts[:, :2], 0) - np.nanmin(pts[:, :2], 0))
        if ext.max() > 0.06:
            continue
        out.append({"n": len(c), "xyz": np.nanmedian(pts, 0).tolist(), "zmax": float(np.nanpercentile(pts[:, 2], 90))})
    return out
