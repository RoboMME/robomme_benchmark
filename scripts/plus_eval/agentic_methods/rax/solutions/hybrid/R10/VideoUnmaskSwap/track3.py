"""Incremental container tracker over RGB-D frames (video then live execution)."""
import numpy as np
import tracker as T
from tracker import load_frame, detect_containers, detect_cubes, fix_count, hungarian_min, top_detections, frame_points, cells

class Tracker:
    def __init__(self, log=print, gate=0.03):
        self.log = log; self.gate = gate
        self.tracks = None; self.content = None; self.vel = None
        self.reacq = 0.15; self.unit = 98.0; self.last_t = None; self.hist = []; self.misses = None
        self.cubes = []

    def init_from_video(self, root, frames):
        """frames: video_demo frames sorted. Detect cubes, init, then track the rest."""
        cube_obs = []; first = None
        f0 = load_frame(root, frames[0])
        nob = T.set_obstacles(f0, frames[0]['state'][:2])
        self.log(f"static obstacle cells: {nob}")
        for k, f in enumerate(frames):
            fr = load_frame(root, f)
            conts = detect_containers(fr)
            if conts and sum(c['n'] for c in conts) > 40:
                first = k; break
            if k % 3 == 0 or k < 3:
                cube_obs.append(detect_cubes(fr))
        if first is None: raise RuntimeError('no containers in video')
        cubes = cube_obs[-1] if cube_obs else []
        for obs in reversed(cube_obs[-6:]):
            if len(obs) > len(cubes): cubes = obs
        self.cubes = cubes
        self.log(f"first container frame {first}; cubes {[(c['color'], [round(v,3) for v in c['xy']]) for c in cubes]}")
        init_k = min(first + 2, len(frames) - 1)
        fr = load_frame(root, frames[init_k])
        groups = fix_count([c['cells'] for c in detect_containers(fr)], len(cubes) if cubes else 0) if cubes else [c['cells'] for c in detect_containers(fr)]
        tracks = np.array([g.mean(0) for g in groups])
        content = [None] * len(tracks)
        if cubes:
            cost = np.linalg.norm(np.array([c['xy'] for c in cubes])[:, None] - tracks[None], axis=2)
            ri, ci = hungarian_min(cost)
            for a, b in zip(ri, ci):
                content[b] = cubes[a]['color']
                if cost[a, b] > 0.05: self.log(f"warn cube {cubes[a]['color']} far ({cost[a,b]:.3f})")
        dets = top_detections(fr)
        if dets:
            self.unit = max(40.0, float(np.median([n for _, n in dets])))
            D = np.array([d for d, _ in dets])
            cost = np.linalg.norm(tracks[:, None] - D[None], axis=2)
            ri, ci = hungarian_min(cost)
            for a, b in zip(ri, ci):
                if cost[a, b] < 0.03 and dets[b][1] > 0.6 * self.unit: tracks[a] = D[b]
        self.tracks = tracks; self.content = content; self.vel = np.zeros_like(tracks)
        self.misses = np.zeros(len(tracks), int); self.consec = np.zeros(len(tracks), int)
        self.last_t = float(frames[init_k].get('source_frame_index', init_k))
        self.log(f"init frame {init_k}: N={len(tracks)} unit={self.unit:.0f} content={content}")
        for f in frames[init_k + 1:]:
            self.update(load_frame(root, f), float(f.get('source_frame_index', 0)))

    def update(self, fr, t, ee=None):
        """fr: loaded frame dict; t: time in source-frame units (20 Hz ticks); ee: robot TCP xyz to mask."""
        dt = max(1.0, t - self.last_t) if self.last_t is not None else 1.0
        self.last_t = t
        dets = top_detections(fr, unit=self.unit)
        if ee is not None and ee[2] < 0.25 and dets:
            dets = [(d, n) for d, n in dets if np.linalg.norm(d - np.asarray(ee[:2])) > 0.06]
        pred = self.tracks + self.vel * dt
        new = pred.copy(); matched = np.zeros(len(pred), bool)
        gate = self.gate * max(1.0, dt / 2.0) ** 0.5
        if dets:
            D = np.array([d for d, _ in dets]); n = np.array([m for _, m in dets])
            cost = np.linalg.norm(pred[:, None] - D[None], axis=2)
            ri, ci = hungarian_min(np.where(cost < gate, cost, 1e3))
            for a, b in zip(ri, ci):
                if cost[a, b] < gate:
                    w = 1.0 if n[b] > 0.6 * self.unit else 0.5
                    new[a] = w * D[b] + (1 - w) * pred[a]; matched[a] = True
        # re-acquire lost tracks with unmatched, well-supported detections
        if dets and not matched.all():
            used = set(int(b) for a, b in zip(ri, ci) if cost[a, b] < gate)
            cand = [j for j in range(len(dets)) if j not in used and n[j] > 0.5 * self.unit
                    and (not matched.any() or np.linalg.norm(new[matched] - D[j], axis=1).min() > 0.045)]
            lost = [i for i in np.where(~matched)[0]]
            if cand and lost:
                C2 = np.linalg.norm(self.tracks[lost][:, None] - D[cand][None], axis=2)
                r2, c2 = hungarian_min(np.where(C2 < self.reacq, C2, 1e3))
                for a, b in zip(r2, c2):
                    if C2[a, b] < self.reacq:
                        i = lost[a]; new[i] = D[cand[b]]; matched[i] = True
                        if self.consec[i] > 0 or C2[a, b] > gate:
                            self.log(f"reacquire track {i} ({self.content[i]}) after {self.consec[i]} misses, jump {C2[a,b]:.3f}")
                        self.vel[i] = 0
        for i in range(len(new)):
            if matched[i]: self.consec[i] = 0
            else:
                self.consec[i] += 1; self.misses[i] += 1
        v_new = (new - self.tracks) / dt
        self.vel = np.where(matched[:, None], 0.5 * self.vel + 0.5 * v_new, 0.8 * self.vel)
        step = np.linalg.norm(new - self.tracks, axis=1)
        self.tracks = new
        self.hist.append(new.copy())
        return step, matched
