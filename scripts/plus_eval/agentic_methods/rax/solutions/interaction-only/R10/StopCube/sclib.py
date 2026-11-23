"""StopCube helpers: decode observations, project pixels, detect cube/target/button, move."""
import base64, io, json, math, time
import numpy as np
from PIL import Image

def decode(result):
    blocks = result.get("content", [])
    meta = json.loads(next(b["text"] for b in blocks if b.get("type") == "text"))
    imgs = {}
    for att in meta.get("attachments", []):
        b = blocks[att["content_index"]]
        if b.get("type") != "image":
            continue
        im = Image.open(io.BytesIO(base64.b64decode(b["data"])))
        arr = np.array(im)
        if att["kind"] == "rgb":
            arr = np.array(im.convert("RGB"))
        imgs[(att["camera"], att["kind"])] = arr
    return meta, imgs

def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def backproject(meta, depth, cam, us, vs):
    """us, vs arrays of pixel coords -> world points (N,3); depth in uint16 mm."""
    K = np.array(meta["camera_intrinsics"][cam]["matrix"])
    pose = meta["camera_poses"][cam]
    R = quat_to_mat(pose["quaternion_wxyz"]); t = np.array(pose["position_m"])
    us = np.asarray(us); vs = np.asarray(vs)
    d = depth[vs.astype(int), us.astype(int)].astype(float) * 0.001
    xc = (us - K[0, 2]) * d / K[0, 0]
    yc = -(vs - K[1, 2]) * d / K[1, 1]
    pc = np.stack([xc, yc, -d], -1)
    return pc @ R.T + t, d

def world_points(meta, imgs, cam="front"):
    depth = imgs[(cam, "depth")]
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    P, d = backproject(meta, depth, cam, us.ravel(), vs.ravel())
    return P.reshape(H, W, 3), d.reshape(H, W)

def masks(rgb):
    a = rgb.astype(int); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1)
    sat = mx - mn
    purple = (B > G + 60) & (R > G + 40)
    gray = (sat < 12) & (mx > 60)
    table = (R - B > 90)
    return purple, gray, table

def detect(meta, imgs, cam="front", ee=None):
    """Color-agnostic detection from RGB-D.
    cube   = largest non-neutral blob whose surface is 0.024-0.06 m high (cube top face)
    target = largest non-neutral blob 0.010-0.022 m high (flat disc), away from the cube
    cap    = neutral gray blob 0.02-0.06 m high (button cap)"""
    from scipy import ndimage
    rgb = imgs[(cam, "rgb")]
    P, d = world_points(meta, imgs, cam)
    valid = d > 0
    a = rgb.astype(int)
    sat = a.max(-1) - a.min(-1); v = a.max(-1)
    z = P[..., 2]
    out = {}
    inws = valid & (P[..., 0] > -0.35) & (P[..., 0] < 0.45) & (np.abs(P[..., 1]) < 0.6)
    neutral = sat < 14
    colored = inws & ~neutral
    if ee is not None and ee[2] < 0.18:
        near = np.linalg.norm(P[..., :2] - np.asarray(ee[:2])[None, None], axis=-1) < 0.07
        colored = colored & ~near
    def largest(mask, min_px):
        if mask.sum() < min_px:
            return None
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < min_px:
            return None
        return lab == k
    cm = largest(colored & (z > 0.024) & (z < 0.06), 6)
    if cm is not None:
        out["cube"] = np.median(P[cm], 0); out["cube_px"] = int(cm.sum())
        vs, us = np.nonzero(cm); out["cube_uv"] = (float(us.mean()), float(vs.mean()))
    capm_pre = largest(inws & neutral & (v > 60) & (v < 190) & (z > 0.02) & (z < 0.06), 4)
    tmask = colored & (z > 0.010) & (z < 0.022)
    if "cube" in out:
        tmask &= np.linalg.norm(P[..., :2] - out["cube"][None, None, :2], axis=-1) > 0.045
    tm = largest(tmask, 6)
    if tm is None:
        # fallback for a neutral-colored target: any flat blob 0.010-0.022 m high away from cube and cap
        fb = inws & (z > 0.010) & (z < 0.022)
        if "cube" in out:
            fb &= np.linalg.norm(P[..., :2] - out["cube"][None, None, :2], axis=-1) > 0.045
        if capm_pre is not None:
            cxy = P[capm_pre][:, :2].mean(0)
            fb &= np.linalg.norm(P[..., :2] - cxy[None, None], axis=-1) > 0.06
        if ee is not None and ee[2] < 0.18:
            fb &= np.linalg.norm(P[..., :2] - np.asarray(ee[:2])[None, None], axis=-1) > 0.07
        tm = largest(fb, 10)
        if tm is not None:
            out["target_fallback"] = True
    if tm is not None:
        ztop = np.percentile(P[tm][:, 2], 90)
        top = tm & (z > ztop - 0.003)
        sel = top if top.sum() >= 4 else tm
        out["target"] = P[sel].mean(0); out["target_px"] = int(tm.sum())
    capm = capm_pre
    if capm is not None:
        out["cap"] = P[capm].mean(0); out["cap_px"] = int(capm.sum())
        out["cap_top"] = float(np.percentile(P[capm][:, 2], 95))
    return out

def observe(ctx):
    return decode(ctx.call("robomme_observe"))

def step(ctx, rows):
    return decode(ctx.call("robomme_step", actions=[list(map(float, r)) for r in rows]))

def clip(v, lim=1.0):
    return max(-lim, min(lim, v))

def move_rows(cur, target, grip=-1.0, vmax=1.0):
    """One action row toward target position."""
    dpos = (np.asarray(target) - np.asarray(cur)) / 0.05
    n = np.linalg.norm(dpos)
    if n > vmax:
        dpos = dpos / n * vmax
    return [dpos[0], dpos[1], dpos[2], 0, 0, 0, grip]
