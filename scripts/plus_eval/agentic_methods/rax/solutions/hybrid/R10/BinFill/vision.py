import base64, io, json
import numpy as np
from PIL import Image


def parse(result):
    """Decode an MCP observe/step result into meta + numpy images."""
    blocks = result.get("content", [])
    meta = None
    for b in blocks:
        if b.get("type") == "text":
            try:
                meta = json.loads(b["text"]); break
            except Exception:
                continue
    imgs = {}
    for att in (meta or {}).get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        if att["kind"] == "rgb":
            imgs[(att["camera"], "rgb")] = np.asarray(im.convert("RGB"))
        else:
            imgs[(att["camera"], "depth")] = np.asarray(im).astype(np.float32) * 0.001
    return meta, imgs


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def world_points(meta, depth, cam):
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_R(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    d = depth
    pc = np.stack([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d], -1)
    pw = pc @ R.T + t
    pw[d <= 0] = np.nan
    return pw


def hsv(rgb):
    im = Image.fromarray(rgb).convert("HSV")
    a = np.asarray(im).astype(np.float32)
    return a[..., 0] * 360 / 255, a[..., 1] / 255, a[..., 2] / 255


def color_masks(rgb):
    H, S, V = hsv(rgb)
    sat = (S > 0.45) & (V > 0.12)
    def hr(lo, hi):
        return (H >= lo) & (H < hi)
    return {
        "red": sat & (hr(0, 12) | hr(345, 361)),
        "green": sat & hr(90, 160),
        "blue": sat & hr(215, 255),
    }


def components(mask, min_px=4):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) >= min_px:
            out.append((ys, xs))
    return out


def label(mask):
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        return lab, n
    except Exception:
        h, w = mask.shape
        lab = np.zeros((h, w), int); n = 0
        for y0 in range(h):
            for x0 in range(w):
                if mask[y0, x0] and lab[y0, x0] == 0:
                    n += 1; st = [(y0, x0)]; lab[y0, x0] = n
                    while st:
                        y, x = st.pop()
                        for yy, xx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and lab[yy, xx] == 0:
                                lab[yy, xx] = n; st.append((yy, xx))
        return lab, n


COLOR_HUES = {"red": 0, "orange": 30, "yellow": 58, "green": 120, "cyan": 180, "blue": 240, "magenta": 300, "purple": 275}


def hue_name(h):
    best, bd = None, 999
    for k, c in COLOR_HUES.items():
        d = abs((h - c + 180) % 360 - 180)
        if d < bd:
            best, bd = k, d
    return best, bd


def square_yaw(xy):
    """Yaw (mod pi/2) of a square footprint from its points: min-area bounding box."""
    xy = xy - xy.mean(0)
    best, ba = 0.0, 1e9
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        c, s = np.cos(a), np.sin(a)
        p = xy @ np.array([[c, -s], [s, c]])
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < ba:
            ba, best = area, a
    # refine
    for a in best + np.deg2rad(np.arange(-2, 2.01, 0.25)):
        c, s = np.cos(a), np.sin(a)
        p = xy @ np.array([[c, -s], [s, c]])
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < ba:
            ba, best = area, a
    return float(best % (np.pi / 2)), float(np.sqrt(ba))


def detect(meta, imgs, cam, table_z=None):
    """Return list of objects (raised blobs) seen by cam in world coords."""
    rgb = imgs[(cam, "rgb")]; dep = imgs[(cam, "depth")]
    P = world_points(meta, dep, cam)
    Z = P[..., 2]
    if table_z is None:
        table_z = float(np.nanpercentile(Z[np.isfinite(Z)], 30))
    H, S, Vv = hsv(rgb)
    valid = np.isfinite(Z) & (np.abs(P[..., 0]) < 1.0) & (np.abs(P[..., 1]) < 1.0)
    raised = valid & (Z > table_z + 0.012) & (Z < table_z + 0.12)
    colored = raised & (S > 0.4) & (Vv > 0.1)
    objs = []
    lab, n = label(colored)
    for i in range(1, n + 1):
        m = lab == i
        cnt = int(m.sum())
        if cnt < 6:
            continue
        hh = H[m]; ang = np.deg2rad(hh)
        hm = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
        name, hd = hue_name(hm)
        pts = P[m]
        ztop = float(np.nanpercentile(pts[:, 2], 90))
        top = pts[pts[:, 2] > ztop - 0.006]
        ys, xs = np.nonzero(m)
        yaw, side = square_yaw(top[:, :2]) if len(top) >= 5 else (0.0, 0.0)
        objs.append(dict(kind="cube", color=name, hue=hm, px=cnt, uv=(float(xs.mean()), float(ys.mean())),
                         xy=top[:, :2].mean(0).tolist(), ztop=ztop, yaw=yaw, side=side, cam=cam))
    # bin: dark low region surrounded by rim
    dark = valid & (Vv < 0.08) & (Z > table_z + 0.005) & (Z < table_z + 0.06)
    lab, n = label(dark)
    bins = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 30:
            continue
        pts = P[m]
        bins.append(dict(kind="bin", px=int(m.sum()), xy=pts[:, :2].mean(0).tolist(), zfloor=float(np.nanmedian(pts[:, 2])),
                         ext=np.ptp(pts[:, :2], 0).tolist(), cam=cam))
    bins.sort(key=lambda b: -b["px"])
    # button: grey raised blob, low saturation, top below 0.045 above table
    grey = raised & (S < 0.3) & (Z < table_z + 0.045)
    lab, n = label(grey)
    btns = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 10:
            continue
        pts = P[m]
        zmx = float(np.nanpercentile(pts[:, 2], 95))
        cap = pts[pts[:, 2] > zmx - 0.006]
        btns.append(dict(kind="grey", px=int(m.sum()), xy=pts[:, :2].mean(0).tolist(), zmax=zmx, capxy=cap[:, :2].mean(0).tolist(),
                         ext=np.ptp(pts[:, :2], 0).tolist(), cam=cam))
    return dict(table_z=table_z, cubes=objs, bins=bins, greys=btns)
