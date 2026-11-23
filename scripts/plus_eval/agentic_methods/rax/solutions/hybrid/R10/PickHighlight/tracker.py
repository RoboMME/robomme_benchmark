"""Multi-cube tracker in a static camera image stream."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from perception import detect_cubes, white_mask

class Tracker:
    def __init__(self, gate=14.0, min_area=25):
        self.tracks = []   # dict(id, uv, vel, area, miss, hl)
        self.gate = gate; self.min_area = min_area
        self.nframes = 0
        self.max_motion = 0.0
        self.log = []

    def init(self, blobs):
        med = np.median([b["area"] for b in blobs]) if blobs else 100
        for b in blobs:
            n = 2 if b["area"] > 1.7 * med else 1   # merged pair at start
            for k in range(n):
                self.tracks.append(dict(id=len(self.tracks), uv=np.array([b["u"], b["v"]]),
                                        vel=np.zeros(2), area=b["area"] / n, miss=0, hl=[]))

    def update(self, img):
        blobs = [b for b in detect_cubes(img) if b["area"] >= self.min_area]
        self.nframes += 1
        if not self.tracks:
            self.init(blobs)
            return 0.0
        med = np.median([t["area"] for t in self.tracks])
        # duplicate large blobs so 2 tracks may share them
        slots = []
        for bi, b in enumerate(blobs):
            slots.append(bi)
            if b["area"] > 1.6 * med:
                slots.append(bi)
        pred = np.array([t["uv"] + t["vel"] for t in self.tracks])
        if slots:
            B = np.array([[blobs[bi]["u"], blobs[bi]["v"]] for bi in slots])
            C = np.linalg.norm(pred[:, None, :] - B[None, :, :], axis=2)
            C = np.where(C > self.gate * 2.5, 1e6, C)
            ri, ci = linear_sum_assignment(C)
        else:
            ri, ci = [], []
        matched = set()
        motion = 0.0
        for i, j in zip(ri, ci):
            if C[i, j] >= 1e6:
                continue
            t = self.tracks[i]; b = blobs[slots[j]]
            new = np.array([b["u"], b["v"]])
            shared = slots.count(slots[j]) > 1 and sum(1 for ii, jj in zip(ri, ci) if C[ii, jj] < 1e6 and slots[jj] == slots[j]) > 1
            if shared:
                # merged blob: keep own prediction (coast) but stay near blob
                newuv = 0.5 * (t["uv"] + t["vel"]) + 0.5 * new
            else:
                newuv = new
            v = newuv - t["uv"]
            t["vel"] = 0.6 * v + 0.4 * t["vel"]
            motion = max(motion, float(np.linalg.norm(v)))
            t["uv"] = newuv; t["miss"] = 0
            if not shared:
                t["area"] = 0.8 * t["area"] + 0.2 * b["area"]
            matched.add(i)
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t["miss"] += 1
                t["uv"] = t["uv"] + t["vel"]; t["vel"] *= 0.7
        self.max_motion = max(self.max_motion, motion)
        return motion

    def score_highlight(self, img):
        import cv2
        wm = white_mask(img)
        if not hasattr(self, "base_white"):
            self.base_white = cv2.dilate(wm, np.ones((5, 5), np.uint8))
        wm = wm & (1 - self.base_white)
        H, W = wm.shape
        vv, uu = np.mgrid[0:H, 0:W]
        for t in self.tracks:
            side = np.sqrt(max(t["area"], 20))
            d = np.hypot(uu - t["uv"][0], vv - t["uv"][1])
            ring = (d <= 1.05 * side) & (d >= 0.72 * side)
            t["hl"].append(float(wm[ring].mean()) if ring.any() else 0.0)
