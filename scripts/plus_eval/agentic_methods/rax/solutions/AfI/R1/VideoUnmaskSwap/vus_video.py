"""Conditioning-video analysis: track containers through the video and find the one
that covers (and then carries) each target-coloured cube."""
import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

import vus_percept as vp
from vus_io import log

GATE = 0.06            # max per-frame association distance (m) (scaled by dt below)
R_ASSOC = 0.04         # container centre within this of the last cube position covers it
MERGE_EXT_RATIO = 1.6  # detection this much larger than a track's usual extent = merged blob
COAST_DECAY = 0.85
MIN_DET_HEIGHT = 0.012


def _assign(cost):
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if _lsa is not None:
        r, c = _lsa(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs, used_r, used_c = [], set(), set()
    for idx in np.argsort(cost, axis=None):
        i, j = divmod(int(idx), m)
        if i in used_r or j in used_c:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
    return pairs


class Track:
    _next = 0

    def __init__(self, det, t):
        self.id = Track._next
        Track._next += 1
        self.pos = np.array(det["top_xy"], dtype=float)
        self.z = float(det["zmax"])
        self.vel = np.zeros(2)
        self.ext = det["ext"]
        self.height = det["height"]
        self.last_t = t
        self.born_t = t
        self.matched = True
        self.n_matched = 1
        self.hist = [(t, self.pos.copy(), "new")]
        self.det = det
        self.color = det.get("color")

    def summary(self):
        return {"id": self.id, "pos": self.pos.round(4).tolist(), "ext": round(self.ext, 4),
                "height": round(self.height, 4), "n_matched": self.n_matched, "color": self.color,
                "born_t": round(self.born_t, 3), "last_t": round(self.last_t, 3)}


class TargetState:
    def __init__(self, color):
        self.color = color
        self.cube_xy = None
        self.cube_z = None
        self.seen = 0
        self.last_seen_t = None
        self.carrier = None
        self.assoc_t = None
        self.events = []
        self.interact = {}


class VideoTracker:
    def __init__(self, colors):
        self.tracks = []
        self.targets = [TargetState(c) for c in colors]
        self.prev_t = None
        self.frames = 0

    def alive(self):
        return [tr for tr in self.tracks if tr.last_t is not None]

    def step(self, t, comps, ee_pos=None):
        self.frames += 1
        dt = 0.05 if self.prev_t is None else max(1e-3, t - self.prev_t)
        self.prev_t = t
        dets = [c for c in comps if not c["robot"] and c["height"] >= MIN_DET_HEIGHT]
        target_cubes = {}
        for ts in self.targets:
            cc = vp.cube_candidates(dets, ts.color)
            target_cubes[ts.color] = cc[0] if cc else None
        cube_ids = {id(c) for c in target_cubes.values() if c is not None}
        cdets = [d for d in dets if id(d) not in cube_ids]
        self._update_tracks(t, dt, cdets, ee_pos)
        for ts in self.targets:
            self._update_target(ts, t, target_cubes.get(ts.color))

    def _split_merged(self, dets, pred, t):
        """Split a blob that contains the predicted positions of >=2 established tracks
        (touching containers) by k-means on its top-face points, seeded at the predictions."""
        out = []
        est = [k for k, tr in enumerate(self.tracks)
               if tr.n_matched >= 5 and t - tr.last_t < 0.6]
        for d in dets:
            pts = d.get("top_pts")
            cx, cy, w, l, yaw = d["fp"]
            inside = [k for k in est
                      if np.linalg.norm(pred[k] - np.array([cx, cy])) < 0.5 * l + 0.005]
            big = len(inside) >= 2 and d["ext"] > 1.3 * max(self.tracks[k].ext for k in inside)
            if not big or pts is None or len(pts) < 10 * len(inside):
                out.append(d)
                continue
            cents = np.array([pred[k] for k in inside], dtype=float)
            for _ in range(8):
                dd = np.linalg.norm(pts[:, None, :] - cents[None, :, :], axis=2)
                a = np.argmin(dd, axis=1)
                for q in range(len(cents)):
                    if (a == q).any():
                        cents[q] = pts[a == q].mean(axis=0)
            dd = np.linalg.norm(pts[:, None, :] - cents[None, :, :], axis=2)
            a = np.argmin(dd, axis=1)
            for q in range(len(cents)):
                sub = pts[a == q]
                if sub.shape[0] < 5:
                    continue
                nd = dict(d)
                nd["top_xy"] = cents[q].copy()
                nd["ext"] = float(max(np.ptp(sub[:, 0]), np.ptp(sub[:, 1])))
                nd["fp"] = (float(cents[q][0]), float(cents[q][1]), nd["ext"], nd["ext"], 0.0)
                nd["split"] = True
                nd["parent"] = d
                nd["top_pts"] = sub
                nd["n"] = int(d["n"] * sub.shape[0] / max(1, len(pts)))
                out.append(nd)
        return out

    def _update_tracks(self, t, dt, dets, ee_pos):
        tracks = self.tracks
        for tr in tracks:
            tr.matched = False
        if tracks and dets:
            pred0 = [tr.pos + tr.vel * dt for tr in tracks]
            dets = self._split_merged(dets, pred0, t)
        if tracks and dets:
            pred = np.array([tr.pos + tr.vel * dt for tr in tracks])
            dpos = np.array([d["top_xy"] for d in dets])
            cost = np.linalg.norm(pred[:, None, :] - dpos[None, :, :], axis=2)
            tz_ = np.array([tr.z for tr in tracks])
            dz_ = np.array([d["zmax"] for d in dets])
            cost = cost + 0.5 * np.abs(tz_[:, None] - dz_[None, :])
            for i, tr in enumerate(tracks):
                for j, d in enumerate(dets):
                    if d["ext"] > MERGE_EXT_RATIO * max(tr.ext, 0.02) and not d.get("split"):
                        cost[i, j] += 0.5  # merged blob: do not let it pull the track
            gate = GATE + 2.0 * dt * 0.1
            used = set()
            for i, j in _assign(cost):
                if cost[i, j] > gate:
                    continue
                tr, d = tracks[i], dets[j]
                newpos = np.array(d["top_xy"], dtype=float)
                v = (newpos - tr.pos) / dt
                tr.vel = 0.5 * v + 0.5 * tr.vel
                sp = np.linalg.norm(tr.vel)
                if sp > 1.0:
                    tr.vel *= 1.0 / sp
                tr.pos = newpos
                tr.z = float(d["zmax"])
                tr.ext = 0.8 * tr.ext + 0.2 * d["ext"] if d["ext"] < MERGE_EXT_RATIO * tr.ext else tr.ext
                tr.height = 0.8 * tr.height + 0.2 * d["height"]
                tr.last_t = t
                tr.matched = True
                tr.n_matched += 1
                tr.det = d
                tr.hist.append((t, tr.pos.copy(), "m"))
                used.add(j)
            # unmatched tracks: merged, carried by the robot, or occluded -> coast
            for tr in tracks:
                if tr.matched:
                    continue
                p = tr.pos + tr.vel * dt
                status = "coast"
                for j, d in enumerate(dets):
                    cx, cy, w, l, yaw = d["fp"]
                    if np.linalg.norm(p - np.array([cx, cy])) < 0.5 * l + 0.01:
                        status = "merged"
                        break
                if ee_pos is not None and status == "coast" and np.linalg.norm(tr.pos - np.asarray(ee_pos[:2])) < 0.06:
                    p = np.asarray(ee_pos[:2], dtype=float)
                    status = "with_ee"
                    tr.vel = np.zeros(2)
                else:
                    tr.vel = tr.vel * COAST_DECAY
                tr.pos = p
                tr.hist.append((t, tr.pos.copy(), status))
            new = [d for j, d in enumerate(dets) if j not in used]
        else:
            new = list(dets)
            for tr in tracks:
                if not tr.matched:
                    tr.vel *= COAST_DECAY
                    tr.pos = tr.pos + tr.vel * dt
                    tr.hist.append((t, tr.pos.copy(), "coast"))
        for d in new:
            if d["n"] < 25:
                continue
            # an unmatched detection close to a coasting track re-acquires it
            best, bd = None, 1e9
            for tr in tracks:
                if tr.matched:
                    continue
                dd = np.linalg.norm(tr.pos - np.array(d["top_xy"]))
                if dd < bd:
                    best, bd = tr, dd
            if best is not None and bd < 0.10 and d["ext"] < MERGE_EXT_RATIO * max(best.ext, 0.02):
                best.pos = np.array(d["top_xy"], dtype=float)
                best.vel = np.zeros(2)
                best.last_t = t
                best.matched = True
                best.n_matched += 1
                best.det = d
                best.hist.append((t, best.pos.copy(), "reacq"))
                continue
            if d["ext"] > 0.25 or d.get("split"):
                continue
            if any(np.linalg.norm(tr.pos - np.array(d["top_xy"])) < 0.04 for tr in tracks):
                continue
            tracks.append(Track(d, t))
        # prune short-lived tracks that have not been matched for a while
        keep = []
        protected = {ts.carrier for ts in self.targets if ts.carrier is not None}
        for tr in tracks:
            if tr.id in protected or tr.n_matched >= 5 or t - tr.last_t < 1.0:
                keep.append(tr)
        self.tracks = keep

    def _update_target(self, ts, t, cube):
        if cube is not None:
            xy = np.array(cube["top_xy"], dtype=float)
            if ts.carrier is not None:
                ts.events.append(("revealed", round(t, 3), xy.round(4).tolist(), ts.carrier))
            ts.cube_xy = xy
            ts.cube_z = float(cube["zmax"])
            ts.seen += 1
            ts.last_seen_t = t
            ts.carrier = None
            return
        if ts.seen == 0:
            return
        if ts.carrier is None:
            best, bd = None, 1e9
            for tr in self.tracks:
                if tr.last_t is None or t - tr.last_t > 0.3:
                    continue
                if ts.cube_z is not None and tr.z < ts.cube_z + 0.008:
                    continue  # a cover must be taller than the cube (rejects stray cube tracks)
                d = np.linalg.norm(tr.pos - ts.cube_xy)
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd < R_ASSOC:
                ts.carrier = best.id
                ts.assoc_t = t
                ts.events.append(("covered", round(t, 3), ts.cube_xy.round(4).tolist(), best.id, round(bd, 4)))
            return
        car = self.get(ts.carrier)
        if car is None:
            return
        for tr in self.tracks:
            if tr.id == ts.carrier:
                continue
            if np.linalg.norm(tr.pos - car.pos) < 0.09:
                ts.interact[tr.id] = ts.interact.get(tr.id, 0) + 1

    def get(self, tid):
        for tr in self.tracks:
            if tr.id == tid:
                return tr
        return None

    def result(self):
        out = []
        for ts in self.targets:
            r = {"color": ts.color, "seen": ts.seen, "last_seen_t": ts.last_seen_t,
                 "cube_xy": None if ts.cube_xy is None else ts.cube_xy.round(4).tolist(),
                 "carrier": ts.carrier, "assoc_t": ts.assoc_t, "events": ts.events[-12:],
                 "interact": ts.interact}
            out.append(r)
        return out


def ranked_candidates(tracker, ts_index, final_dets):
    """Ordered list of (track, reason) for the target, best first, restricted to tracks
    that are matched to a detection in the final (live) frame where possible."""
    ts = tracker.targets[ts_index]
    live = [tr for tr in tracker.tracks if tr.matched]
    order = []
    car = tracker.get(ts.carrier) if ts.carrier is not None else None
    if car is not None:
        if car.matched:
            order.append((car, "carrier"))
        else:
            # carrier lost in the final frame: nearest live track to its last position
            near = sorted(live, key=lambda tr: np.linalg.norm(tr.pos - car.pos))
            if near:
                order.append((near[0], "carrier_nearest_live"))
    elif ts.cube_xy is not None:
        near = sorted(live, key=lambda tr: np.linalg.norm(tr.pos - ts.cube_xy))
        if near:
            order.append((near[0], "nearest_last_cube_xy"))
    # alternatives: tracks that interacted with the carrier, then the rest by size
    inter = sorted(ts.interact.items(), key=lambda kv: -kv[1])
    for tid, _ in inter:
        tr = tracker.get(tid)
        if tr is not None and tr.matched and all(tr is not o for o, _ in order):
            order.append((tr, "interacted"))
    rest = [tr for tr in live if all(tr is not o for o, _ in order)]
    rest.sort(key=lambda tr: -tr.height)
    for tr in rest:
        order.append((tr, "other"))
    return order
