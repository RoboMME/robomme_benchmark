"""PickHighlight controller (v2: wrist-only identity tracking, front camera = shuffle alarm only,
timing = last shuffle end + min quiet gap; see memory/PROGRESS.md "v2").

Task: "first press the button, then pick up all highlighted cubes one by one, finally press the
button to stop". Observed (exploration ep1):
  * white discs appear under some cubes ~steps 6-54 (the highlight), then they vanish;
  * from then on, roughly every ~141 steps, cubes swap places along curved paths for ~40 steps;
  * lifting a non-highlighted cube ends the episode as a failure.
Method: park the wrist camera at a viewing pose above the cubes, track every cube's identity with
dense observations (1-2 steps per call), record which identities carried discs, and do each
manipulation (button press / pick-lift-replace) inside a quiet window between shuffles.

Transition table (phase -> exits):
  INIT      -> WATCH (cubes found)                     | fail-safe: keep observing
  WATCH     -> PRESS1 (quiet window long enough, button not yet pressed, highlight frozen)
            -> PICK  (quiet window long enough, pending highlighted cube)
            -> PRESS2 (all highlighted picked, quiet window)
            -> WATCH (otherwise: keep observing)
  PRESS1/2  -> RETURN (done or aborted)
  PICK      -> RETURN (lifted+replaced -> mark picked | aborted before grasp)
  RETURN    -> WATCH (re-associate tracks at viewing pose)
  Episode end at any time -> exit.
"""
import math
import time
import traceback

import numpy as np

import robot as R
from percep import detect_cubes, detect_button, disc_score, cube_yaw
from tracker import Tracker

VERSION = "pickhl-v2"

# ---- tunables (tested ranges: ep1 only unless stated) ----
PERIOD_DEFAULT = 95        # shuffle period guess before 2 starts seen (ep1 ~141, ep2 95-106 -> use the short one)
SHUFFLE_LEN = 45           # steps a shuffle lasts (ep1: ~40)
QUIET_CONFIRM = 4          # steps without motion to declare a shuffle over
MOTION_EPS = 0.005         # m between updates counted as motion
QGAP_DEFAULT = 62          # quiet steps from a shuffle's last motion to the next start (ep2: 66-78)
OVERDUE_GIVEUP = 220       # if no shuffle seen this long after the predicted start, stop waiting for one
GRASP_TO_RELEASE = 22      # steps from closing the gripper to having released the cube (ep2: ~19)
REGION_MARGIN = 0.08       # detections outside the initial cube/button bbox + margin are ignored
FRONT_MIN_PX = 30          # front-camera blobs smaller than this are partial (occluded) cubes
FRONT_ALARM_DRIFT = 0.05   # front-camera shadow tracks drifting this far (>=2 tracks) = shuffle during manipulation
SAFETY = 5                 # steps margin before predicted next shuffle
HL_THRESH = 0.45           # disc score threshold (ep1: highlighted 0.6-1.0, others 0.0)
VIEW_Z_MIN, VIEW_Z_MAX = 0.28, 0.52
CAM_OFF = np.array([0.046, 0.020, 0.068])   # wrist camera position minus TCP at yaw 0 (ep1 home)
GRASP_Z = 0.018
PLACE_Z = 0.024
LIFT_Z = 0.09
APPROACH_Z = 0.08
OPEN, CLOSE = -1.0, 1.0


def log(*a):
    print("[%s]" % VERSION, *a, flush=True)


class Ctl:
    def __init__(self, ctx):
        self.rb = R.Robot(ctx)
        self.trk = None
        self.button = None
        self.view = None
        self.hl_frozen = False
        self.discs_seen = False
        self.baseline = None
        self.pressed1 = False
        self.pressed2 = False
        self.motion_hist = []          # (step, moving bool)
        self.shuffle_starts = []
        self.in_shuffle = False
        self.in_motion = False
        self.motion_onset = None
        self.quiet_ref = None
        self.shuffle_ends = []
        self.last_motion_step = -999
        self.pick_attempts = {}
        self.manip_target = None
        self.region = None
        self.extra_presses = 0
        self.last_press_step = -999
        self.shadow = None
        self.manip_ref = None
        self.alarm = False
        self.t0 = time.time()

    # ---------------- perception ----------------
    def wrist_dets(self):
        rgb, dep, K, pose = self.rb.cam("wrist")
        if rgb is None or dep is None:
            return [], None, None
        cs, Wm = detect_cubes(rgb, dep, K, pose)
        return self.in_region(cs), Wm, rgb

    def in_region(self, cs):
        if self.region is None:
            return cs
        lo, hi = self.region
        return [c for c in cs if lo[0] <= c["xy"][0] <= hi[0] and lo[1] <= c["xy"][1] <= hi[1]]

    def front_dets(self):
        rgb, dep, K, pose = self.rb.cam("front")
        if rgb is None or dep is None:
            return [], None, None
        cs, Wm = detect_cubes(rgb, dep, K, pose)
        return self.in_region(cs), Wm, rgb

    def at_view(self):
        return self.view is not None and np.linalg.norm(self.rb.pos - self.view) < 0.03

    def update_tracking(self, source="wrist", frozen=None):
        """Update tracker from the current observation; returns moving flag.
        v2: the front camera (used while the arm is away from the viewing pose) is too noisy for identity
        tracking (arm occlusion -> centroid jumps up to 6 cm, ep2). It only feeds a shadow tracker that raises
        an alarm on large coherent drift; the real tracks stay frozen and shuffle starts are only recorded
        from the wrist view."""
        if source != "wrist":
            return self.update_shadow(frozen)
        cs, Wm, rgb = self.wrist_dets()
        if Wm is None:
            return False
        dets = [c["xy"] for c in cs]
        disp, nm = self.trk.update(dets, self.rb.step_id, frozen=frozen)
        moving = disp > MOTION_EPS
        # highlight discs (only from wrist top-down view, before the highlight is frozen)
        if source == "wrist" and not self.hl_frozen:
            bxy = None if self.button is None else self.button["xy"]
            rel = np.array([disc_score(rgb, Wm, p, exclude_xy=bxy)[0] for p in self.trk.pos])
            any_disc = bool((rel > HL_THRESH).any())
            if any_disc:
                self.discs_seen = True
                self.trk.hl_score = np.maximum(self.trk.hl_score, rel)
                self.trk.hl_frames += (rel > HL_THRESH).astype(int)
            elif self.discs_seen:
                self.hl_frozen = True
                log("highlight ended at step %d; highlighted ids %s scores %s frames %s" % (
                    self.rb.step_id, list(np.flatnonzero(self.trk.hl_score > HL_THRESH)),
                    np.round(self.trk.hl_score, 2).tolist(), self.trk.hl_frames.tolist()))
        # shuffle bookkeeping: a shuffle is confirmed when >=2 tracks moved >2 cm from their quiet
        # reference positions; its start is back-dated to the first motion after the quiet period.
        s = self.rb.step_id
        if self.quiet_ref is None:
            self.quiet_ref = self.trk.pos.copy()
        if moving:
            if not self.in_motion:
                self.motion_onset = s
            self.in_motion = True
            self.last_motion_step = s
        drift = np.linalg.norm(self.trk.pos - self.quiet_ref, axis=1)
        if not self.in_shuffle and int((drift > 0.02).sum()) >= 2:
            self.in_shuffle = True
            st = self.motion_onset if self.motion_onset is not None else s
            self.shuffle_starts.append(st)
            log("shuffle start ~%d confirmed at %d (starts %s)" % (st, s, self.shuffle_starts))
        if self.in_motion and s - self.last_motion_step >= QUIET_CONFIRM:
            self.in_motion = False
            self.motion_onset = None
            if self.in_shuffle:
                self.in_shuffle = False
                self.shuffle_ends.append(self.last_motion_step)
                log("shuffle over at %d (last motion %d); tracks %s" % (s, self.last_motion_step, self.trk.summary()))
            self.quiet_ref = self.trk.pos.copy()
        return moving

    def update_shadow(self, frozen=None):
        if self.shadow is None:
            return False
        cs, Wm, rgb = self.front_dets()
        if Wm is None:
            return False
        # partially occluded cubes give small blobs with shifted centroids -> ignore them
        cs = [c for c in cs if c["n"] >= FRONT_MIN_PX]
        self.shadow.vel[:] = 0          # no extrapolation: cubes move <= ~4 cm between updates
        self.shadow.update([c["xy"] for c in cs], self.rb.step_id, frozen=frozen)
        self.shadow.vel[:] = 0
        drift = np.linalg.norm(self.shadow.pos - self.manip_ref, axis=1)
        drift = np.where(self.shadow.last_matched, drift, 0.0)
        if frozen is not None:
            drift = np.where(frozen, 0.0, drift)
        big = drift > FRONT_ALARM_DRIFT
        persist = big & getattr(self, "_prev_big", np.zeros_like(big))
        self._prev_big = big
        nbig = int(persist.sum())
        if nbig >= 2 and not self.alarm:
            self.alarm = True
            log("ALARM: front shadow drift at %d: tracks %s drift %s" % (
                self.rb.step_id, list(np.flatnonzero(drift > FRONT_ALARM_DRIFT)), np.round(drift[drift > FRONT_ALARM_DRIFT], 3).tolist()))
        return nbig >= 2

    def begin_manip(self, target=None):
        self.manip_target = target
        self.shadow = self.trk.copy()
        self.shadow.vel[:] = 0
        self.manip_ref = self.trk.pos.copy()
        self._prev_big = np.zeros(self.trk.n, bool)
        self.alarm = False

    def quiet_gap(self):
        """Conservative quiet gap between a shuffle's last motion and the next start (wrist-observed only).
        ep2: 66-78 steps while start-to-start varied 85-106 (shuffle durations 21-35)."""
        g = []
        for st in self.shuffle_starts:
            prev = [e for e in self.shuffle_ends if e < st]
            if prev and 30 < st - prev[-1] < 300:
                g.append(st - prev[-1])
        if not g:
            return QGAP_DEFAULT
        g.sort()
        return float(g[0]) if len(g) < 4 else float(g[1])

    def period(self):
        """Conservative (minimum) observed interval between shuffle starts.
        ep1: ~141 steps; ep2: 95-106 steps -> the period varies per episode."""
        st = self.shuffle_starts
        if len(st) >= 2:
            d = sorted(b - a for a, b in zip(st[:-1], st[1:]) if 60 < b - a < 260)
            if d:
                return float(d[0]) if len(d) < 4 else float(d[1])
        return PERIOD_DEFAULT

    def steps_to_next_shuffle(self):
        """Steps until the next shuffle may start: last wrist-observed shuffle end + quiet gap.
        0 while a shuffle is in progress or overdue (a shuffle may have been missed while the arm was away:
        then wait at the viewing pose until the next one is observed)."""
        s = self.rb.step_id
        if self.in_shuffle or self.in_motion:
            return 0
        if not self.shuffle_ends:
            if not self.shuffle_starts:
                return None
            return 0
        last_end = self.shuffle_ends[-1]
        nxt = last_end + self.quiet_gap()
        if nxt < s:
            if s - nxt > OVERDUE_GIVEUP:
                return None
            return 0
        return nxt - s

    # ---------------- motions ----------------
    def go_view(self):
        self.manip_target = None
        self.mv(self.view, yaw=0.0, grip=OPEN, tol=0.01, vmax=0.02)
        self.shadow_done = self.shadow
        self.shadow = None

    def watch_step(self, n=2):
        self.rb.step([[0, 0, 0, 0, 0, 0, OPEN]] * n)
        return self.update_tracking("wrist")

    def manip_cb(self):
        """Called after every robot call during manipulation: keep tracking with the front camera."""
        fz = np.zeros(self.trk.n, bool)
        if self.manip_target is not None:
            fz[self.manip_target] = True
        try:
            self.update_tracking("front", frozen=fz)
        except Exception:
            log("manip_cb error:\n" + traceback.format_exc())

    def mv(self, target, **kw):
        kw.setdefault("verbose", False)
        return R.move_to(self.rb, target, max_rows=4, callback=self.manip_cb, **kw)

    def press(self, tag):
        rb = self.rb
        s0 = rb.step_id
        self.begin_manip(None)
        b = self.button["xy"]
        self.mv([b[0], b[1], 0.10], yaw=0.0, grip=CLOSE, tol=0.008, verbose=False)
        # refine cap centre from the wrist camera right above
        rgb, dep, K, pose = rb.cam("wrist")
        bb = detect_button(rgb, dep, K, pose) if rgb is not None else None
        if bb is not None and np.linalg.norm(bb["xy"] - b) < 0.03:
            b = 0.5 * (b + bb["xy"])
        self.mv([b[0], b[1], 0.05], yaw=0.0, grip=CLOSE, tol=0.005, verbose=False)
        zs = []
        for i in range(14):
            rb.step([[0, 0, -0.12, 0, 0, 0, CLOSE]] * 2)
            self.manip_cb()
            zs.append(rb.pos[2])
            if len(zs) >= 3 and abs(zs[-1] - zs[-2]) < 0.0008 and zs[-1] < 0.032:
                break
            if zs[-1] < 0.012:
                break
        rb.step([[0, 0, -0.05, 0, 0, 0, CLOSE]] * 3)
        self.manip_cb()
        zmin = rb.pos[2]
        log("%s: pressed at xy %s, z min %.4f, steps %d" % (tag, np.round(b, 3), zmin, rb.step_id - s0))
        self.last_press_step = rb.step_id
        self.mv([b[0], b[1], 0.10], yaw=0.0, grip=CLOSE, tol=0.01, verbose=False)
        return zmin

    def pick(self, i):
        """Pick track i, lift, put back. Returns 'done', 'abort' or 'fail'."""
        rb, trk = self.rb, self.trk
        s0 = rb.step_id
        xy = trk.pos[i].copy()
        self.begin_manip(i)
        self.mv([xy[0], xy[1], APPROACH_Z], yaw=0.0, grip=OPEN, tol=0.008, verbose=False)
        cs, Wm, rgb = self.wrist_dets()
        cand = [c for c in cs if np.linalg.norm(c["xy"] - xy) < 0.035]
        if not cand:
            log("pick %d: no cube within 3.5cm of track pos %s; abort" % (i, np.round(xy, 3)))
            return "abort"
        c = min(cand, key=lambda c: np.linalg.norm(c["xy"] - xy))
        # nearest other track must be clearly farther than the matched detection (identity check)
        others = [np.linalg.norm(trk.pos[j] - c["xy"]) for j in range(trk.n) if j != i]
        if others and min(others) < 0.03:
            log("pick %d: ambiguous identity (other track %.3f m away); abort" % (i, min(others)))
            return "abort"
        yaw, npx = cube_yaw(Wm, rgb, c["xy"])
        yaw = 0.0 if yaw is None else yaw
        g = c["xy"]
        log("pick %d: target %s yaw %.1f deg" % (i, np.round(g, 3), math.degrees(yaw)))
        self.mv([g[0], g[1], 0.04], yaw=yaw, grip=OPEN, tol=0.008, verbose=False)
        self.mv([g[0], g[1], GRASP_Z], yaw=yaw, grip=OPEN, vmax=0.012, tol=0.004, verbose=False)
        # last chance to abort: a shuffle starting now would swap cubes under us
        k = self.steps_to_next_shuffle()
        if self.alarm or self.shuffle_imminent(margin=GRASP_TO_RELEASE):
            log("pick %d: shuffle imminent before grasp (alarm %s, next in %s); abort" % (i, self.alarm, k))
            self.mv([g[0], g[1], APPROACH_Z], yaw=yaw, grip=OPEN, tol=0.01, verbose=False)
            return "abort"
        R.hold(rb, 6, CLOSE, callback=self.manip_cb)
        self.mv([g[0], g[1], LIFT_Z], yaw=yaw, grip=CLOSE, tol=0.008, verbose=False)
        f = float(rb.state[6:8].sum())
        held = 0.025 < f < 0.05
        log("pick %d: lifted z=%.3f fingers=%.4f held=%s" % (i, rb.pos[2], f, held))
        self.mv([g[0], g[1], PLACE_Z + 0.015], yaw=yaw, grip=CLOSE, tol=0.006, verbose=False)
        self.mv([g[0], g[1], PLACE_Z], yaw=yaw, grip=CLOSE, vmax=0.01, tol=0.004, verbose=False)
        R.hold(rb, 5, OPEN, callback=self.manip_cb)
        self.mv([g[0], g[1], PLACE_Z + 0.03], yaw=yaw, grip=OPEN, tol=0.012, verbose=False)
        log("pick %d: %s, steps %d (crit est %d)" % (i, "done" if held else "missed", rb.step_id - s0, self.pick_crit(i)))
        trk.pos[i] = g
        return "done" if held else "fail"

    def pick_crit(self, i):
        """Estimated steps from pick start until the cube is placed back and released."""
        tgt = np.array([self.trk.pos[i][0], self.trk.pos[i][1], APPROACH_Z])
        d = float(np.linalg.norm(tgt - self.rb.pos))
        return int(d / 0.022) + 30   # ep2: ~39 steps start->release from the viewing pose

    def press_crit(self):
        b = self.button["xy"]
        d = float(np.linalg.norm(np.array([b[0], b[1], 0.10]) - self.rb.pos))
        return int(d / 0.018) + 4 + 22

    def shuffle_imminent(self, margin=SAFETY):
        k = self.steps_to_next_shuffle()
        return k is not None and k < margin

    # ---------------- main logic ----------------
    def init(self):
        rb = self.rb
        rb.observe()
        log("instruction:", rb.meta.get("instruction"), "limits", rb.meta.get("episode_limits"))
        cs, Wm, rgb = self.wrist_dets()
        fcs, _, _ = self.front_dets()
        pts = [c["xy"] for c in cs]
        if len(fcs) > len(cs):
            log("front sees more cubes (%d) than wrist (%d)" % (len(fcs), len(cs)))
        self.trk = Tracker(pts)
        bw = detect_button(*rb.cam("wrist"))
        bf = detect_button(*rb.cam("front"))
        bs = [b["xy"] for b in (bw, bf) if b is not None]
        self.button = dict(xy=np.mean(bs, 0)) if bs else None
        log("init: %d cubes; button %s" % (self.trk.n, None if not bs else np.round(self.button["xy"], 3)))
        allp = np.array(pts + ([self.button["xy"]] if bs else []))
        lo, hi = allp.min(0), allp.max(0)
        if fcs:  # the front camera sees the whole table; widen the region by its detections
            fp = np.array([c["xy"] for c in fcs])
            self.region = (np.minimum(lo, fp.min(0)) - REGION_MARGIN, np.maximum(hi, fp.max(0)) + REGION_MARGIN)
        else:
            self.region = (lo - REGION_MARGIN, hi + REGION_MARGIN)
        log("region", np.round(self.region, 3).tolist())
        ctr = (lo + hi) / 2
        half = float(max(hi - lo) / 2)
        hcam = (half + 0.07) / 0.9
        z = float(np.clip(hcam - CAM_OFF[2], VIEW_Z_MIN, VIEW_Z_MAX))
        self.view = np.array([ctr[0] - CAM_OFF[0], ctr[1] - CAM_OFF[1], z])
        log("view pose", np.round(self.view, 3))

    def run(self):
        rb = self.rb
        if self.trk is not None:
            self.loop()
            return
        self.init()
        # move to the viewing pose in small chunks while tracking
        for _ in range(12):
            if self.at_view():
                break
            d = self.view - rb.pos
            n = 3
            dp = d / max(1.0, np.linalg.norm(d) / (0.018 * n)) / (n * R.ALPHA) / R.POS_SCALE
            rb.step([list(np.clip(dp, -1, 1)) + [0, 0, 0, OPEN]] * n)
            self.update_tracking("wrist")
        R.move_to(rb, self.view, yaw=0.0, grip=OPEN, tol=0.01, verbose=False)
        self.loop()

    def loop(self):
        rb = self.rb
        while not rb.ended:
            self.update_tracking("wrist")
            pending = [i for i in range(self.trk.n) if self.trk.hl_score[i] > HL_THRESH and not self.trk.picked[i]]
            k = self.steps_to_next_shuffle()
            quiet = (not self.in_shuffle) and (not self.in_motion)
            # fallbacks: highlight never ended / no shuffle seen for a long time
            if not self.hl_frozen and self.discs_seen and rb.step_id > 200:
                self.hl_frozen = True; log("highlight forced frozen at %d" % rb.step_id)
            if not self.discs_seen and rb.step_id > 300:
                log("no highlight discs seen by step %d; giving up" % rb.step_id)
                return
            seen_shuffle = bool(self.shuffle_starts) or rb.step_id - max(self.last_motion_step, 0) > 160
            ready = self.hl_frozen and quiet and seen_shuffle
            if ready and self.button is None:
                self.button_tries = getattr(self, "button_tries", 0) + 1
                bs = [b["xy"] for b in (detect_button(*rb.cam("wrist")), detect_button(*rb.cam("front"))) if b is not None]
                if bs:
                    self.button = dict(xy=np.mean(bs, 0)); log("button found late at", np.round(self.button["xy"], 3))
                elif self.button_tries > 20 and not self.pressed1:
                    log("button never found; skipping presses"); self.pressed1 = True
            if ready and self.button is not None and not self.pressed1:
                if k is None or k > self.press_crit() + SAFETY:
                    log("PRESS1 at step %d (next shuffle in %s)" % (rb.step_id, k))
                    self.press("press1")
                    self.pressed1 = True
                    self.go_view(); self.reassociate()
                    continue
            elif ready and self.pressed1 and pending:
                # fewest attempts first (a cube that keeps failing must not block the others), then nearest
                i = min(pending, key=lambda j: (self.pick_attempts.get(j, 0),
                                                np.linalg.norm(self.trk.pos[j] - rb.pos[:2])))
                if k is None or k > self.pick_crit(i) + SAFETY:
                    log("PICK track %d at step %d (next shuffle in %s) %s" % (i, rb.step_id, k, self.trk.summary()))
                    res = self.pick(i)
                    if res == "done":
                        self.trk.picked[i] = True
                    self.pick_attempts[i] = self.pick_attempts.get(i, 0) + 1
                    self.go_view(); self.reassociate()
                    continue
            elif ready and self.pressed1 and not pending and not self.pressed2 and self.button is not None:
                if k is None or k > self.press_crit() + SAFETY:
                    log("PRESS2 at step %d" % rb.step_id)
                    self.press("press2")
                    self.pressed2 = True
                    self.go_view(); self.reassociate()
                    continue
            elif self.pressed2 and not rb.ended and self.extra_presses < 3 and ready and self.button is not None \
                    and rb.step_id - self.last_press_step > 60 and (k is None or k > self.press_crit() + SAFETY):
                # episode should have ended with press2; maybe a press did not register -> press again (max 3)
                self.extra_presses += 1
                log("episode still running after press2 at step %d; extra press %d" % (rb.step_id, self.extra_presses))
                self.press("press%d" % (2 + self.extra_presses))
                self.go_view(); self.reassociate()
                continue
            # default: observe (dense during shuffles / near predicted start)
            n = 1 if (self.in_shuffle or (k is not None and k < 4)) else 2
            self.watch_step(n)

    def reassociate(self):
        """After returning to the viewing pose: match tracks to detections.
        If the front alarm fired (cubes may have moved while away), start from the shadow positions."""
        sh = getattr(self, "shadow_done", None)
        alarm = self.alarm
        if alarm and sh is not None:
            self.trk.pos = sh.pos.copy()
        self.trk.vel[:] = 0
        cs, Wm, rgb = self.wrist_dets()
        dets = [c["xy"] for c in cs]
        disp, nm = self.trk.update(dets, self.rb.step_id, gate=0.06 if alarm else 0.04)
        self.trk.vel[:] = 0
        # recovery: a single lost track + a single unexplained detection -> re-attach
        lost = np.flatnonzero(~self.trk.last_matched)
        left = self.trk.last_unused
        if 1 <= len(lost) <= 2 and len(left) == len(lost):
            for i in lost:
                dd = np.linalg.norm(left - self.trk.pos[i], axis=1)
                j = int(np.argmin(dd))
                if dd[j] < 0.15:
                    log("reassociate: re-attach lost track %d %s -> %s" % (i, np.round(self.trk.pos[i], 3), np.round(left[j], 3)))
                    self.trk.pos[i] = left[j]; self.trk.coast[i] = 0
                    left = np.delete(left, j, axis=0)
        self.quiet_ref = self.trk.pos.copy()
        self.alarm = False
        log("reassociate at %d: matched %d/%d, max disp %.3f; %s" % (self.rb.step_id, nm, self.trk.n, disp, self.trk.summary()))


def main(ctx):
    c = Ctl(ctx)
    t0 = time.time()
    for attempt in range(3):
        try:
            c.run()
            break
        except R.EpisodeEnded:
            log("episode ended at step %d (task_complete=%s)" % (c.rb.step_id, (c.rb.meta or {}).get("transition", {}).get("task_complete")))
            break
        except Exception:
            log("exception:\n" + traceback.format_exc())
            if c.rb.ended or time.time() - t0 > 1000:
                break
            try:
                c.rb.observe()
                if c.view is not None:
                    c.go_view()
            except Exception:
                log("recovery failed:\n" + traceback.format_exc())
    log("main exit at step %d, wall %.1fs" % (c.rb.step_id, time.time() - t0))
