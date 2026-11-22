"""World-frame (table xy) multi-cube tracker fed by wrist + front RGB-D detections."""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
from perception import hsv, blobs, white_mask

def band_detections(meta, img, depth, cam, deproject_fn, min_area=12):
    P = deproject_fn(meta, depth, cam)
    z = P[..., 2]
    S = hsv(img)[..., 1].astype(int)
    m = ((z > 0.012) & (z < 0.075) & (depth > 0) & (S >= 45)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bl, lab = blobs(m, min_area=min_area, max_area=20000)
    H, W = depth.shape
    out = []
    for b in bl:
        sel = lab == b["label"]
        pts = P[sel]
        top = pts[pts[:, 2] > pts[:, 2].max() - 0.008]
        x0, y0, w, h = b["bbox"]
        edge = x0 <= 1 or y0 <= 1 or x0 + w >= W - 1 or y0 + h >= H - 1
        xy = np.median(top[:, :2], 0)
        ext = np.ptp(top[:, :2], 0).max() if len(top) > 2 else 0
        yaw = None
        if len(top) >= 6:
            (cx, cy), (w_, h_), ang = cv2.minAreaRect((top[:, :2] * 1000.0).astype(np.float32))
            yaw = float((np.deg2rad(ang) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        out.append(dict(xy=xy, area=b["area"], u=b["u"], v=b["v"], cam=cam, edge=edge, ext=float(ext), yaw=yaw,
                        ztop=float(np.median(top[:, 2])), n=int(sel.sum())))
    return out

class WorldTracker:
    def __init__(self, gate=0.025):
        self.allow_spawn = True
        self.tracks = []  # dict(id, xy, vel, miss, hl, seen)
        self.gate = gate
        self.frame = 0
        self.max_motion = 0.0
        self.last_motion = 0.0

    def detections(self, meta, imgs, deproject_fn, cams=("wrist", "front")):
        dets = []
        for cam in cams:
            if (cam, "depth") not in imgs:
                continue
            d = band_detections(meta, imgs[(cam, "rgb")], imgs[(cam, "depth")], cam, deproject_fn)
            for x in d:
                # single-cube sized blobs only (top-face extent <= ~5cm), no partial edge blobs
                if x["edge"] or x["ext"] > 0.075 or x["ext"] < 0.015:
                    continue
                dets.append(x)
        # dedupe: prefer wrist
        out = []
        for x in sorted(dets, key=lambda x: 0 if x["cam"] == "wrist" else 1):
            if all(np.linalg.norm(x["xy"] - y["xy"]) > 0.02 for y in out):
                out.append(x)
        return out

    def init(self, dets):
        for x in dets:
            self.tracks.append(dict(id=len(self.tracks), xy=x["xy"].copy(), vel=np.zeros(2), miss=0, hl=[], seen=1))

    def update(self, meta, imgs, deproject_fn, cams=("wrist", "front")):
        dets = self.detections(meta, imgs, deproject_fn, cams)
        self.frame += 1
        if not self.tracks:
            self.init(dets)
            return 0.0
        if not dets:
            for t in self.tracks:
                t["miss"] += 1
            return 0.0
        pred = np.array([t["xy"] + t["vel"] for t in self.tracks])
        D = np.array([x["xy"] for x in dets])
        C = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
        Cg = np.where(C > self.gate, 1e3, C)
        ri, ci = linear_sum_assignment(Cg)
        matched = set(); motion = 0.0
        for i, j in zip(ri, ci):
            if Cg[i, j] >= 1e3:
                continue
            t = self.tracks[i]
            v = dets[j]["xy"] - t["xy"]
            # per-frame motion normalised by steps since last seen
            step_v = v / (1 + t["miss"])
            t["vel"] = 0.5 * step_v + 0.5 * t["vel"]
            motion = max(motion, float(np.linalg.norm(step_v)))
            t["xy"] = dets[j]["xy"].copy(); t["miss"] = 0; t["seen"] += 1
            if dets[j]["cam"] == "wrist" and dets[j].get("yaw") is not None:
                t["yaw"] = dets[j]["yaw"]
            matched.add(i)
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t["miss"] += 1
                t["vel"] = t["vel"] * 0.5
        if self.allow_spawn:
            used = set(int(j) for i, j in zip(ri, ci) if Cg[i, j] < 1e3)
            for j, x in enumerate(dets):
                if j in used:
                    continue
                if all(np.linalg.norm(x["xy"] - t["xy"]) > 0.03 for t in self.tracks):
                    self.tracks.append(dict(id=len(self.tracks), xy=x["xy"].copy(), vel=np.zeros(2), miss=0, hl=[], seen=1))
        self.last_motion = motion
        self.max_motion = max(self.max_motion, motion)
        self._last_dets = dets
        return motion

    def score_highlight(self, meta, imgs, project_fn, cam="wrist"):
        img = imgs[(cam, "rgb")]
        wm = white_mask(img)
        if not hasattr(self, "base_white"):
            self.base_white = cv2.dilate(wm, np.ones((5, 5), np.uint8))
        wm = wm & (1 - self.base_white)
        H, W = wm.shape
        vv, uu = np.mgrid[0:H, 0:W]
        for t in self.tracks:
            u, v = project_fn(meta, cam, np.array([t["xy"][0], t["xy"][1], 0.04]))
            u0, v0 = project_fn(meta, cam, np.array([t["xy"][0] + 0.04, t["xy"][1], 0.04]))
            side = max(4.0, float(np.hypot(u0 - u, v0 - v)))  # 4cm in px
            d = np.hypot(uu - u, vv - v)
            ring = (d <= 1.05 * side) & (d >= 0.72 * side)
            t["hl"].append(float(wm[ring].mean()) if ring.any() else 0.0)
