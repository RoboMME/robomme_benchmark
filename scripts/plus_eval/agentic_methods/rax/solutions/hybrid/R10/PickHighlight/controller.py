"""PickHighlight controller v6 (v5 + wide-gate reacquire after excursions).
Observed task behaviour (exploration): white rings mark the target cubes for ~25-40 steps
after reset; then every ~100 steps the cubes shuffle for ~40 steps, for the whole episode.
Strategy: track every cube in table xy from a *static* camera pose (wrist + front RGB-D);
freeze tracks while the arm is away; leave the tracking pose only right after a shuffle
ends, for one short excursion (press button / pick-lift-replace one highlighted cube)."""
import os, sys, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from robot import Robot, EpisodeEnded, move_to, hold, ee_yaw, rot_err, deproject, project, log
from perception import white_mask, blobs
from wtracker import WorldTracker

OUT = "/workspace/output"
GRASP_Z = 0.022
LIFT_Z = 0.12
PLACE_Z = 0.03
TRACK_Z = 0.30
CAM_OFF = np.array([0.046, 0.020])  # wrist camera xy offset from TCP at yaw 0
MOTION_THR = 0.0015                 # m/step: cube motion => shuffle in progress
MAX_STEP_CAP = 2000

def save_json(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1)
    except Exception as e:
        log("save_json fail", e)

class State:
    def __init__(self):
        self.tr = WorldTracker()
        self.scoring = True
        self.tracking = True       # update tracks only while camera parked
        self.motion_log = []
        self.last_motion_step = None
        self.shuffle_starts = []
        self._prev_cam = None
        self._in_shuffle = False
        self.reacq = False
        self.pose_since = 0

    def on_step(self, r):
        cp = np.array(r.meta["camera_poses"]["wrist"]["position_m"])
        prev = self._prev_cam; self._prev_cam = cp
        static = prev is not None and np.linalg.norm(cp - prev) < 0.0015
        if not (self.tracking and static):
            return
        if self.reacq:
            # first static frame after an excursion: re-associate with a wide gate,
            # do not count the jump as a shuffle
            g = self.tr.gate; self.tr.gate = 0.06
            try:
                self.tr.update(r.meta, r.imgs, deproject, ("wrist", "front"))
                for t in self.tr.tracks:
                    t["vel"] = np.zeros(2)
            finally:
                self.tr.gate = g
            self.reacq = False
            self.pose_since = r.step_id
            return
        mo = self.tr.update(r.meta, r.imgs, deproject, ("wrist", "front"))
        self.motion_log.append((r.step_id, round(mo * 1000, 1)))
        if mo > MOTION_THR:
            if not self._in_shuffle:
                self.shuffle_starts.append(r.step_id)
            self._in_shuffle = True
            self.last_motion_step = r.step_id
        elif self.last_motion_step is not None and r.step_id - self.last_motion_step >= 3:
            self._in_shuffle = False
        if self.scoring:
            self.tr.score_highlight(r.meta, r.imgs, project)

    def still_for(self, r):
        if self.last_motion_step is None:
            return None
        return r.step_id - self.last_motion_step

def fast_move(r, target, yaw, grip, vmax=1.0, tol=0.006, rtol=0.06, max_steps=30):
    target = np.asarray(target, float)
    for _ in range(max_steps):
        if r.ended:
            return False
        s = r.state
        dp = target - s[:3]; dr = rot_err(s, yaw)
        if np.linalg.norm(dp) < tol and np.linalg.norm(dr) < rtol:
            return True
        n = np.linalg.norm(dp)
        a = dp / 0.05
        if n > vmax * 0.05:
            a = a * (vmax * 0.05 / n)
        b = np.clip(dr / 0.5, -0.6, 0.6)
        r.step([list(a) + list(b) + [grip]])
    return False

def find_button(r):
    img = r.imgs[("wrist", "rgb")]; dep = r.imgs[("wrist", "depth")]
    P = deproject(r.meta, dep, "wrist")
    wb, lab = blobs(white_mask(img), 20, 3000)
    cands = []
    for b in wb:
        sel = (lab == b["label"]) & (dep > 0)
        pts = P[sel]
        if len(pts) < 10:
            continue
        z = np.median(pts[:, 2])
        if z > 0.08 or z < 0.003:
            continue
        x0, y0, w, h = b["bbox"]
        reg = P[y0:y0 + h, x0:x0 + w].reshape(-1, 3)
        reg = reg[(reg[:, 2] > 0.005) & (reg[:, 2] < 0.06)]
        c = np.median(reg[:, :2], 0) if len(reg) else np.median(pts[:, :2], 0)
        ext = float(np.ptp(pts[:, :2], 0).max())
        if ext > 0.12:
            continue
        cands.append(dict(xy=c, area=b["area"], ext=ext))
    cands.sort(key=lambda c: -c["area"])
    log("button cands", [(np.round(c["xy"], 3).tolist(), c["area"], round(c["ext"], 3)) for c in cands])
    return cands[0] if cands else None

def press_button(r, bxy, tag):
    bx, by = bxy
    s0 = r.step_id
    fast_move(r, [bx, by, 0.07], 0, 1, tol=0.008, rtol=0.1, max_steps=30)
    zs = []; last = r.state[2]
    for i in range(14):
        e = np.array([bx, by]) - r.state[:2]
        r.step([[e[0] / 0.05, e[1] / 0.05, -0.4, 0, 0, 0, 1]])
        z = float(r.state[2]); zs.append(round(z, 4))
        if z < 0.03 and last - z < 0.0015:
            break
        last = z
    hold(r, 2, 1)
    r.save(tag + "_pressed")
    fast_move(r, [bx, by, 0.10], 0, 1, tol=0.01, rtol=0.2, max_steps=6)
    log("press", tag, "z trace", zs, "step", r.step_id, "took", r.step_id - s0)
    return zs

def choose_yaw(t, tracks):
    base = t.get("yaw")
    base = 0.0 if base is None else base
    best = None
    for yaw in [base, base + np.pi / 2 if base < 0 else base - np.pi / 2]:
        d = np.array([-np.sin(yaw), np.cos(yaw)])  # finger closing axis
        clear = 1.0
        for o in tracks:
            if o is t:
                continue
            for sgn in (1, -1):
                clear = min(clear, np.linalg.norm(o["xy"] - (t["xy"] + sgn * 0.045 * d)))
        score = clear - 0.02 * abs(yaw)
        if best is None or score > best[0]:
            best = (score, yaw, clear)
    return best[1], best[2]

def pick_excursion(r, S, t, tag):
    """Fast grasp-lift-replace of track t. Tracks are frozen meanwhile."""
    s0 = r.step_id
    gx, gy = t["xy"]
    yaw, clear = choose_yaw(t, S.tr.tracks)
    log(tag, "track", t["id"], "at", np.round([gx, gy], 3).tolist(), "yaw", round(np.rad2deg(yaw), 1), "clear", round(clear, 3), "step", s0)
    fast_move(r, [gx, gy, 0.075], yaw, -1, tol=0.006, rtol=0.05, max_steps=30)
    fast_move(r, [gx, gy, GRASP_Z], yaw, -1, vmax=0.8, tol=0.004, rtol=0.05, max_steps=10)
    prev_w = None
    for i in range(10):
        r.step([[0, 0, 0, 0, 0, 0, 1]])
        w = float(r.state[6] + r.state[7])
        if prev_w is not None and abs(prev_w - w) < 0.0005 and i >= 3:
            break
        prev_w = w
    fast_move(r, [gx, gy, LIFT_Z], yaw, 1, tol=0.01, rtol=0.2, max_steps=8)
    if r.ended:
        return False
    hold(r, 1, 1)
    w = float(r.state[6] + r.state[7])
    held = 0.02 < w < 0.075 and r.state[2] > 0.08
    r.save(f"{tag}_lifted")
    log(tag, "lift width", round(w, 4), "z", round(float(r.state[2]), 3), "held", held, "step", r.step_id, "took", r.step_id - s0)
    # set it back down where it was
    fast_move(r, [gx, gy, PLACE_Z], yaw, 1, vmax=0.8, tol=0.006, rtol=0.2, max_steps=8)
    for i in range(4):
        r.step([[0, 0, 0, 0, 0, 0, -1]])
    fast_move(r, [gx, gy, 0.08], yaw, -1, tol=0.01, rtol=0.3, max_steps=5)
    t["xy"] = np.array([gx, gy]); t["vel"] = np.zeros(2)
    log(tag, "replaced; excursion steps", r.step_id - s0)
    return held

def go_track_pose(r, S, pose):
    S.tracking = False
    fast_move(r, pose, 0, -1, tol=0.006, rtol=0.05, max_steps=30)
    hold(r, 1, -1)
    S._prev_cam = None
    S.reacq = True
    S.tracking = True
    hold(r, 3, -1)

def wait_window(r, S, limit, min_still=4, max_fresh=14, long_still=80):
    """Wait at the tracking pose until just after a shuffle ends (or a long quiet spell)."""
    s0 = r.step_id
    while not r.ended and r.step_id < limit:
        st = S.still_for(r)
        lm = S.last_motion_step if S.last_motion_step is not None else -10**6
        quiet_since = r.step_id - max(lm, S.pose_since)
        if st is not None and min_still <= st <= max_fresh and lm >= S.pose_since:
            return "fresh"
        if quiet_since >= long_still and r.step_id - S.pose_since >= long_still:
            return "quiet"
        r.step([[0, 0, 0, 0, 0, 0, -1]])
    return "limit"

def run(ctx):
    global OUT
    t0 = time.time()
    try:
        os.makedirs(OUT, exist_ok=True)
    except Exception:
        import tempfile
        OUT = tempfile.mkdtemp()
    r = Robot(ctx, outdir=OUT)
    r.observe()
    log("start step", r.step_id, "state", np.round(r.state, 3).tolist(), r.meta.get("instruction"))
    limit = r.step_id + r.remaining() - 5
    S = State(); S.on_step(r); r.on_step = S.on_step
    S.pose_since = r.step_id
    btn = find_button(r)
    # ---- watch: highlight + first shuffle, at the reset pose
    base_white = int(white_mask(r.imgs[("wrist", "rgb")]).sum())
    hl_on = hl_off = None
    while r.step_id < 400 and not r.ended:
        r.step([[0, 0, 0, 0, 0, 0, -1]])
        wsum = int(white_mask(r.imgs[("wrist", "rgb")]).sum())
        if wsum > base_white + 150:
            hl_on = hl_on or r.step_id; hl_off = None
        elif hl_on is not None and hl_off is None:
            hl_off = r.step_id
        st = S.still_for(r)
        if hl_off is not None and S.shuffle_starts and st is not None and st >= 4 \
                and r.step_id >= S.shuffle_starts[0] + 30:
            break
        if hl_off is not None and not S.shuffle_starts and r.step_id > hl_off + 150:
            break
    S.scoring = False; S.tr.allow_spawn = False
    tracks = S.tr.tracks
    sc = np.array([np.mean(sorted(t["hl"])[-10:]) if t["hl"] else 0.0 for t in tracks])
    mx = float(sc.max()) if len(sc) else 0.0
    thr = max(0.15, 0.35 * mx)
    hl = [t for t, s in zip(tracks, sc) if s > thr]
    log("watch end step", r.step_id, "hl", hl_on, hl_off, "shuffles", S.shuffle_starts, "tracks", len(tracks),
        "scores", np.round(sc, 2).tolist(), "highlighted", [t["id"] for t in hl], "t", round(time.time() - t0, 1))
    save_json("watch.json", dict(hl_on=hl_on, hl_off=hl_off, scores=sc, thr=thr, hl=[t["id"] for t in hl],
                                 tracks=[dict(id=t["id"], xy=t["xy"]) for t in tracks]))
    if btn is None:
        btn = find_button(r)
    bxy = btn["xy"] if btn else None
    # tracking pose: wrist camera above the cube cluster centroid
    cen = np.mean([t["xy"] for t in tracks], 0) if tracks else r.state[:2]
    pose = np.array([cen[0] - CAM_OFF[0], cen[1] - CAM_OFF[1], TRACK_Z])
    # ---- press 1 (right after the first shuffle)
    S.tracking = False
    if bxy is not None:
        press_button(r, bxy, "press1")
    else:
        log("NO BUTTON FOUND")
    go_track_pose(r, S, pose)
    S.pose_since = r.step_id
    # ---- picks, one per still window
    remaining = list(hl); done = []
    fails = {}
    while remaining and not r.ended and r.step_id < limit - 120:
        why = wait_window(r, S, limit - 120)
        if why == "limit" or r.ended:
            break
        # nearest highlighted track to the current TCP
        i = int(np.argmin([np.linalg.norm(t["xy"] - r.state[:2]) for t in remaining]))
        t = remaining[i]
        S.tracking = False
        ok = pick_excursion(r, S, t, f"pick{len(done)}")
        if r.ended:
            break
        if ok:
            remaining.pop(i); done.append(t["id"])
        else:
            fails[t["id"]] = fails.get(t["id"], 0) + 1
            if fails[t["id"]] >= 3:
                remaining.pop(i)
        go_track_pose(r, S, pose)
        S.pose_since = r.step_id
        log("picked", done, "remaining", [x["id"] for x in remaining], "step", r.step_id, "why", why, "t", round(time.time() - t0, 1))
    # ---- final press
    if bxy is not None and not r.ended:
        press_button(r, bxy, "press2")
    if not r.ended:
        hold(r, 3, -1)
    save_json("motion.json", dict(motion=S.motion_log, shuffles=S.shuffle_starts))
    log("END step", r.step_id, "ended", r.ended, "complete", r.complete, "shuffles", S.shuffle_starts, "t", round(time.time() - t0, 1))

def main(ctx):
    try:
        run(ctx)
    except EpisodeEnded:
        log("episode ended during an action (normal termination)")
    except Exception:
        log("CONTROLLER EXCEPTION", traceback.format_exc())
