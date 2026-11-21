"""Perception for PickXtimes from RGB-D observations."""
import math
import numpy as np
import robo

WS = dict(xmin=-0.42, xmax=0.4, ymin=-0.45, ymax=0.45)

def _inws(P):
    return (P[..., 0] > WS["xmin"]) & (P[..., 0] < WS["xmax"]) & (P[..., 1] > WS["ymin"]) & (P[..., 1] < WS["ymax"])

def world_cloud(meta, cam):
    return robo.deproject_mask(meta, cam, np.ones((256, 256), bool)).reshape(256, 256, 3) if (meta["depth_m"][cam] > 0).all() else _cloud_masked(meta, cam)

def _cloud_masked(meta, cam):
    P = np.full((256, 256, 3), np.nan)
    m = meta["depth_m"][cam] > 0
    pts = robo.deproject_mask(meta, cam, m)
    vs, us = np.nonzero(m)
    P[vs, us] = pts
    return P

COLORS = ("green", "red", "blue", "yellow")

def mask_color(rgb, color):
    h, s, v = robo.hsv(rgb)
    if color == "red":
        return ((h < 10) | (h > 350)) & (s > 0.75) & (v > 0.3)
    if color == "blue":
        return (h > 210) & (h < 250) & (s > 0.5) & (v > 0.2)
    if color == "yellow":
        return (h > 45) & (h < 70) & (s > 0.6) & (v > 0.4)
    return (h > 90) & (h < 150) & (s > 0.5) & (v > 0.25)

def find_cube(meta, cams=("wrist", "front"), color="green"):
    """Return dict(xy, ztop, yaw, n, cam) of the largest green blob, or None."""
    best = None
    for cam in cams:
        rgb = meta["rgb"].get(cam)
        if rgb is None:
            continue
        P = world_cloud(meta, cam)
        mk = mask_color(rgb, color) & _inws(P) & (P[..., 2] > -0.01) & (P[..., 2] < 0.25)
        comps = robo.components(mk, 6)
        if not comps:
            continue
        n, (vs, us) = comps[0]
        pts = P[vs, us]
        ztop = float(np.percentile(pts[:, 2], 90))
        top = pts[pts[:, 2] > ztop - 0.006]
        if len(top) < 4:
            top = pts
        xy = np.median(top[:, :2], 0)
        yaw = None
        try:
            import cv2
            (cx, cy), (w, h), ang = cv2.minAreaRect(top[:, :2].astype(np.float32))
            a = math.radians(ang)
            yaw = (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
        except Exception:
            yaw = rect_yaw(top[:, :2])
        cand = dict(xy=xy, ztop=ztop, yaw=yaw, n=n, cam=cam, ntop=len(top))
        if best is None or (cam == "wrist" and n >= 30) :
            best = cand
            if cam == "wrist" and n >= 30:
                break
    return best

def rect_yaw(xy):
    """Dominant edge orientation of a square footprint, in (-pi/4, pi/4]."""
    if len(xy) < 5:
        return 0.0
    c = xy - xy.mean(0)
    best, ba = None, 0.0
    for a in np.linspace(-math.pi / 4, math.pi / 4, 46, endpoint=False):
        ca, sa = math.cos(a), math.sin(a)
        u = c[:, 0] * ca + c[:, 1] * sa
        v = -c[:, 0] * sa + c[:, 1] * ca
        area = (np.percentile(u, 97) - np.percentile(u, 3)) * (np.percentile(v, 97) - np.percentile(v, 3))
        if best is None or area < best:
            best, ba = area, a
    return float(ba)

def find_target(meta, cams=("wrist", "front")):
    for cam in cams:
        rgb = meta["rgb"].get(cam)
        if rgb is None:
            continue
        P = world_cloud(meta, cam)
        mk = robo.mask_target(rgb) & _inws(P) & (P[..., 2] < 0.03)
        comps = robo.components(mk, 6)
        if not comps:
            continue
        # merge all components within 6 cm of the largest (rings)
        n0, (vs0, us0) = comps[0]
        c0 = np.nanmedian(P[vs0, us0][:, :2], 0)
        allp = []
        for n, (vs, us) in comps:
            p = P[vs, us]
            if np.linalg.norm(np.nanmedian(p[:, :2], 0) - c0) < 0.06:
                allp.append(p)
        pts = np.concatenate(allp)
        xy = (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2
        return dict(xy=xy, z=float(np.median(pts[:, 2])), n=len(pts), cam=cam)
    return None

def find_button(meta, cams=("wrist", "front"), exclude=()):
    """Raised (z 1.2-8 cm) low-saturation blob on the table: the button."""
    for cam in cams:
        rgb = meta["rgb"].get(cam)
        if rgb is None:
            continue
        P = world_cloud(meta, cam)
        h, s, v = robo.hsv(rgb)
        mk = _inws(P) & (P[..., 2] > 0.012) & (P[..., 2] < 0.08) & (s < 0.25)
        comps = robo.components(mk, 15)
        for n, (vs, us) in comps:
            pts = P[vs, us]
            xy = np.median(pts[:, :2], 0)
            if any(np.linalg.norm(xy - np.asarray(e)) < 0.05 for e in exclude):
                continue
            ext = pts[:, :2].max(0) - pts[:, :2].min(0)
            if ext.max() > 0.12:
                continue
            ztop = float(np.percentile(pts[:, 2], 95))
            return dict(xy=xy, ztop=ztop, n=n, cam=cam, ext=ext)
    return None
