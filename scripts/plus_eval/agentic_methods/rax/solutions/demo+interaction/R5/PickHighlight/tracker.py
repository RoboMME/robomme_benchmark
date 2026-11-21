import numpy as np
from percep import detect, cube_pose, highlight_votes

def detections(meta, imgs, cams=("front", "wrist")):
    """fused cube top-centre detections (xy) from given cameras"""
    det = []
    for cam in cams:
        if (cam, "depth") not in imgs:
            continue
        for c in detect(meta, imgs, cam)["cubes"]:
            if c["n"] < 15:
                continue
            ctr, zt, yaw, ext = cube_pose(c["pts"])
            if max(ext) > 0.05 or not (0.028 < zt < 0.055):
                continue
            det.append((np.asarray(ctr, float), cam, c["n"]))
    # fuse duplicates across cameras (prefer wrist when close, it is usually sharper)
    fused = []
    for p, cam, n in det:
        for f in fused:
            if np.linalg.norm(f["xy"] - p) < 0.015:
                f["obs"].append(p); f["xy"] = np.mean(f["obs"], 0); break
        else:
            fused.append({"xy": p.copy(), "obs": [p]})
    return [f["xy"] for f in fused]

class Tracker:
    def __init__(self, log=print):
        self.tracks = []   # dict: xy, v, last_seen(step), id
        self.log = log
        self.step = 0
        self.max_move_hist = []

    def init(self, dets, step):
        self.tracks = [{"id": i, "xy": d.copy(), "v": np.zeros(2), "seen": step, "votes": 0.0} for i, d in enumerate(dets)]
        self.step = step

    def update(self, dets, step, allow_new=False, gate=0.03, frozen_ids=()):
        dt = max(1, step - self.step)
        self.step = step
        pred = [t["xy"] + t["v"] * dt for t in self.tracks]
        pairs = []
        for i, p in enumerate(pred):
            if self.tracks[i]["id"] in frozen_ids:
                continue
            for j, d in enumerate(dets):
                dd = np.linalg.norm(p - d)
                if dd < gate * max(1, min(dt, 3)):
                    pairs.append((dd, i, j))
        pairs.sort()
        ti, dj = set(), set()
        maxmove = 0.0
        for dd, i, j in pairs:
            if i in ti or j in dj:
                continue
            ti.add(i); dj.add(j)
            t = self.tracks[i]
            newv = (dets[j] - t["xy"]) / dt
            mv = np.linalg.norm(dets[j] - t["xy"])
            maxmove = max(maxmove, mv)
            t["v"] = 0.5 * t["v"] + 0.5 * newv if mv > 0.002 else 0.5 * t["v"]
            t["xy"] = dets[j].copy(); t["seen"] = step
        for i, t in enumerate(self.tracks):
            if i not in ti:
                t["v"] = t["v"] * 0.5   # coast, decay velocity
                t["xy"] = t["xy"] + t["v"] * dt * 0.5
        if allow_new:
            for j, d in enumerate(dets):
                if j in dj:
                    continue
                if all(np.linalg.norm(t["xy"] - d) > 0.035 for t in self.tracks):
                    nid = max([t["id"] for t in self.tracks] + [-1]) + 1
                    self.tracks.append({"id": nid, "xy": d.copy(), "v": np.zeros(2), "seen": step, "votes": 0.0})
                    self.log("tracker: new track", nid, np.round(d, 3).tolist(), "step", step)
        self.max_move_hist.append((step, maxmove))
        return maxmove

    def get(self, tid):
        for t in self.tracks:
            if t["id"] == tid:
                return t
        return None
