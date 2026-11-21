"""Multi-cube tracker in world XY with constant-velocity prediction and colour cost."""
import numpy as np

from percep import assign

GATE_M = 0.055          # max association distance (m) between prediction and detection
REACQ_M = 0.10          # re-acquisition radius for tracks missing for a while
LOST_M = 0.30           # during a shuffle: max jump when re-attaching a lost track to a free detection
QUIET_LOST_M = 0.20     # no shuffle: a lost cube can only be where it was (or where a shuffle left it)
COLOR_W = 0.5           # metres of cost per unit chroma distance
COLOR_GATE = 0.10       # chroma distance above which a match is forbidden
MOVE_SPEED = 0.0022     # m/step: a matched track faster than this counts as moving
MOVE_SLOW = 0.0010      # m/step: slower motion counts only when the track has left its anchor
MOVE_DISP = 0.008       # m from anchor for slow motion to count
ANCHOR_SPEED = 0.0007   # m/step: anchor follows a track only when it is this still
MOVER_DIST = 0.012      # m from shuffle-start anchor: track is part of the shuffle cycle
PIN_AFTER = 10          # steps after shuffle start: non-movers are pinned to their anchor
PIN_TOL = 0.012         # m: pinned track accepts detections within this of its anchor


class Track:
    def __init__(self, tid, det, step):
        self.id = tid
        self.xy = np.asarray(det["xy"], float).copy()
        self.vel = np.zeros(2)
        self.color = np.asarray(det["color"], float).copy()
        self.last_step = step
        self.missing = 0
        self.hl_sum = 0.0
        self.hl_n = 0
        self.hl_max = 0.0
        self.highlight = False
        self.done = False
        self.held = False
        self.attempts = 0
        self.moved_recent = 0.0
        self.det = det
        self.seen_xy = self.xy.copy()
        self.seen_step = step
        self.anchor = self.xy.copy()   # position before the current shuffle
        self.mover = False
        self.pin_far = 0
        self.history = [(step, float(self.xy[0]), float(self.xy[1]))]

    def predict(self, step):
        if self.pinned:
            return self.anchor.copy()
        dt = max(0, min(step - self.last_step, 8))
        return self.xy + self.vel * dt

    pinned = False

    def brief(self):
        return dict(id=self.id, xy=[round(float(self.xy[0]), 4), round(float(self.xy[1]), 4)],
                    hl=self.highlight, done=self.done, held=self.held, miss=self.missing,
                    hl_mean=round(self.hl_sum / max(self.hl_n, 1), 3))


class Tracker:
    def __init__(self):
        self.tracks = []
        self.n_init = 0
        self.last_moving = []   # ids moving at last update
        self.frames = 0
        self.shuffle_t0 = None

    def init(self, dets, step):
        self.tracks = [Track(i, d, step) for i, d in enumerate(dets)]
        self.n_init = len(self.tracks)

    def begin_shuffle(self, t0):
        """Called when a shuffle is detected (t0 = estimated start)."""
        self.shuffle_t0 = t0
        for t in self.tracks:
            t.mover = t.held or float(np.linalg.norm(t.xy - t.anchor)) > MOVER_DIST
            t.pinned = False
            t.pin_far = 0

    def end_shuffle(self):
        self.shuffle_t0 = None
        for t in self.tracks:
            t.pinned = False
            t.mover = False
            t.anchor = t.xy.copy()

    def _shuffle_bookkeeping(self, step):
        if self.shuffle_t0 is None:
            for t in self.tracks:
                if t.missing == 0 and float(np.linalg.norm(t.vel)) < ANCHOR_SPEED:
                    t.anchor = t.xy.copy()
            return
        for t in self.tracks:
            if t.held:
                continue
            if not t.mover and not t.pinned:
                if float(np.linalg.norm(t.xy - t.anchor)) > MOVER_DIST:
                    t.mover = True
                elif step - self.shuffle_t0 >= PIN_AFTER:
                    t.pinned = True
                    t.xy = t.anchor.copy()
                    t.vel = np.zeros(2)

    def predictions(self, step, include_held=False):
        return [t.predict(step) for t in self.tracks if include_held or not t.held]

    def active(self):
        return [t for t in self.tracks if not t.held]

    def update(self, dets, step, ignore_near=None, ignore_r=0.0):
        """Associate detections; returns (matches dict tid->det, moving ids)."""
        self.frames += 1
        act = self.active()
        pred = np.array([t.predict(step) for t in act]).reshape(-1, 2)
        D = np.array([d["xy"] for d in dets]).reshape(-1, 2)
        matches = {}
        moving = []
        if len(act) and len(dets):
            dist = np.linalg.norm(pred[:, None, :] - D[None, :, :], axis=2)
            col = np.array([[np.linalg.norm(t.color - d["color"]) for d in dets] for t in act])
            cost = dist + COLOR_W * col
            big = 1e3
            gate = np.where(np.array([t.missing for t in act])[:, None] > 2, REACQ_M, GATE_M)
            cost = np.where((dist <= gate) & (col <= COLOR_GATE), cost, big)
            # pinned (static during a shuffle) tracks claim the detection at their anchor first
            pin_rows = [r for r, t in enumerate(act) if t.pinned]
            claimed = {}
            for r in sorted(pin_rows, key=lambda r: float(dist[r].min())):
                c = int(np.argmin(np.where(np.isin(np.arange(len(dets)), list(claimed.values())), 9.0, dist[r])))
                if dist[r, c] <= PIN_TOL and col[r, c] <= COLOR_GATE and c not in claimed.values():
                    claimed[r] = c
            for r in pin_rows:
                cost[r, :] = big
            for r, c in claimed.items():
                cost[:, c] = big
                cost[r, c] = 0.0
            used_d = set()
            for r, c in assign(cost):
                if cost[r, c] >= big:
                    continue
                t = act[r]
                d = dets[c]
                used_d.add(c)
                newxy = np.asarray(d["xy"], float)
                if t.pinned:
                    far = float(np.linalg.norm(newxy - t.anchor)) > PIN_TOL
                    t.pin_far = t.pin_far + 1 if (far and not d.get("merged")) else 0
                    if t.pin_far >= 2:
                        # it really moves: release the pin
                        t.pinned = False
                        t.mover = True
                    else:
                        t.last_step = step
                        t.missing = 0
                        if not far:
                            t.det = d
                        matches[t.id] = d
                        used_d.add(c)
                        continue
                t.coast = getattr(t, "coast", 0) + 1 if (d.get("merged") and t.mover and t.missing == 0) else 0
                if t.coast and t.coast <= 6:
                    # overlapping cubes: trust constant-velocity coasting over the biased split
                    newxy = t.predict(step)
                dt = max(1, step - t.seen_step)
                v = (newxy - t.seen_xy) / dt
                speed = float(np.linalg.norm(v))
                if t.missing == 0:
                    t.vel = 0.5 * t.vel + 0.5 * v
                elif t.missing <= 4:
                    t.vel = v
                else:
                    t.vel = np.zeros(2)
                t.seen_xy = newxy.copy()
                t.seen_step = step
                disp = float(np.linalg.norm(newxy - t.anchor))
                if t.missing == 0 and (speed > MOVE_SPEED or (speed > MOVE_SLOW and disp > MOVE_DISP)):
                    near = False
                    if ignore_near is not None:
                        near = np.linalg.norm(newxy - np.asarray(ignore_near)[:2]) < ignore_r
                    if not near:
                        moving.append(t.id)
                t.xy = newxy
                if not d.get("merged") and speed < MOVE_SPEED:
                    t.color = 0.9 * t.color + 0.1 * np.asarray(d["color"], float)
                t.last_step = step
                t.missing = 0
                t.det = d
                if len(t.history) < 4000:
                    t.history.append((step, float(newxy[0]), float(newxy[1])))
                matches[t.id] = d
        # re-acquire long-lost tracks with leftover detections (cube count is conserved)
        if len(act) and len(dets):
            lost = [r for r, t in enumerate(act) if t.id not in matches and t.missing >= 2 and not t.pinned]
            free = [c for c in range(len(dets)) if c not in used_d]
            if lost and free:
                lim = LOST_M if self.shuffle_t0 is not None else QUIET_LOST_M
                sub = np.array([[dist[r, c] + COLOR_W * col[r, c] if (dist[r, c] <= lim and col[r, c] <= COLOR_GATE) else 1e3
                                 for c in free] for r in lost])
                for i, j in assign(sub):
                    if sub[i, j] >= 1e3:
                        continue
                    t = act[lost[i]]
                    d = dets[free[j]]
                    used_d.add(free[j])
                    t.xy = np.asarray(d["xy"], float).copy()
                    t.seen_xy = t.xy.copy()
                    t.seen_step = step
                    t.vel = np.zeros(2)
                    t.last_step = step
                    t.missing = 0
                    t.det = d
                    t.reacq = getattr(t, "reacq", 0) + 1
                    if self.shuffle_t0 is None:
                        t.anchor = t.xy.copy()
                    matches[t.id] = d
        for t in act:
            if t.id not in matches:
                t.missing += 1
                # coast with constant velocity for a few updates, then damp
                dt = max(1, step - t.last_step)
                if t.missing <= 6:
                    t.xy = t.xy + t.vel * min(dt, 4)
                if t.missing > 3:
                    t.vel *= 0.6
                t.last_step = step
        self._shuffle_bookkeeping(step)
        self.last_moving = moving
        return matches, moving

    def nearest(self, xy, exclude=None):
        best = None
        for t in self.tracks:
            if exclude is not None and t.id == exclude:
                continue
            if t.held:
                continue
            d = float(np.linalg.norm(t.xy - np.asarray(xy)[:2]))
            if best is None or d < best[0]:
                best = (d, t)
        return best
