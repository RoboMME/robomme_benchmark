"""PickHighlight controller v1.
Detect highlighted cubes (white discs), press button, track every cube through
shuffles at every control step (front+wrist fusion), pick each highlighted cube,
put it back, press button again."""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
from lib import Obs, observe, aa_to_R, R_to_aa, ee_yaw, log
from percep import square_yaw

VERSION = "v5"
HOME = np.array([-0.06, 0.0, 0.30])   # arm above all cubes in front view
POS_GAIN = 1.9   # measured: arm achieves ~0.44 of commanded delta per step (max ~2 cm/step)
ROT_GAIN = 1.5
TOP_Z_MIN = 0.036      # top-face points of resting cubes (top at 0.042)
TOP_Z_MAX = 0.060
ASSIGN_R = 0.035
GRASP_DZ = 0.022
HOVER_Z = 0.13
MOVE_THR = 0.0025       # per-step centre motion counted as shuffle motion
# shuffle period varies per episode (ep2: 100 steps, rehearsal2: ~80); measured online
SHUFFLE_LEN = 40        # observed 32-38
NEED_PICK = 30          # steps from window start to grasp closed (~23 measured) + slack
NEED_PLACE = 25         # steps to release (~20 measured)
LAG_MARGIN = 6          # shuffle onset detection lag
NEED_PRESS = 30
OUT = "/workspace/output"


class Abort(Exception):
    pass


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.obs = None
        self.C = None          # Kx2 centers
        self.hl = None
        self.done = None
        self.held = -1
        self.hist = []         # (step, centers)
        self.trk = open(os.path.join(OUT, "track.jsonl"), "a")
        self.t0 = time.time()
        self.starts = []
        self.period = None
        self.phase = None
        self.base_n = None
        self.prev_ee = None
        self.move_run = 0
        self.pre_quiet = 0
        self.quiet_run = 0
        self.last_move = -100
        self.pick_info = None
        self.fails = 0
        self.pressed_first = False

    # ---------------- perception
    def top_points(self, o, cams=("front", "wrist")):
        pts = []
        for cam in cams:
            if cam not in o.depth:
                continue
            if cam == "wrist" and o.pose["wrist"][0][2] < 0.08:
                continue
            P = o.cloud(cam)
            m = (P[..., 2] > TOP_Z_MIN) & (P[..., 2] < TOP_Z_MAX) & (o.depth[cam] > 0)
            m &= (P[..., 0] > -0.45) & (P[..., 0] < 0.25) & (np.abs(P[..., 1]) < 0.40)
            pts.append(P[m][:, :2])
        X = np.concatenate(pts) if pts else np.zeros((0, 2))
        if getattr(self, "bxy", None) is not None and len(X):
            X = X[np.linalg.norm(X - self.bxy[None], axis=1) > 0.035]
        ee = o.state[:3]
        if ee[2] < 0.15 and len(X):
            X = X[np.linalg.norm(X - ee[None, :2], axis=1) > 0.045]
        return X

    def white_points(self, o):
        """Highlight evidence: front pixels that changed vs the step-0 baseline, near table level."""
        P = o.cloud("front")
        rgb = o.rgb["front"].astype(int)
        drgb = np.abs(rgb - self.base_rgb).sum(-1)
        dd = self.base_depth - o.depth["front"]
        m = ((drgb > 60) | (dd > 0.004)) & (P[..., 2] < 0.03) & (P[..., 2] > -0.01)
        W = P[m][:, :2]
        if getattr(self, 'bxy', None) is not None and len(W):
            W = W[np.linalg.norm(W - self.bxy[None], axis=1) > 0.05]
        return W

    def init_cubes(self, o):
        X = self.top_points(o, ("wrist",))
        if len(X) < 200:
            X = self.top_points(o)
        # grid components at 5 mm
        g = np.floor(X / 0.005).astype(int); g -= g.min(0)
        img = np.zeros(g.max(0) + 3, bool); img[g[:, 0] + 1, g[:, 1] + 1] = True
        img = ndimage.binary_closing(img, iterations=1)
        lab, n = ndimage.label(img)
        C = []
        for i in range(1, n + 1):
            sel = lab[g[:, 0] + 1, g[:, 1] + 1] == i
            pts = X[sel]
            if len(pts) < 30:
                continue
            k = max(1, int(round(np.ptp(pts, 0).max() / 0.045)))
            ext = np.ptp(pts, 0)
            area_k = int(round(len(pts) / max(1, np.median([1]))))
            # split by kmeans while cluster too large
            from scipy.cluster.vq import kmeans2
            for kk in range(1, 5):
                if kk == 1:
                    cs = pts.mean(0)[None]; lb = np.zeros(len(pts), int)
                else:
                    cs, lb = kmeans2(pts, kk, minit="++", seed=0)
                ext = max(np.ptp(pts[lb == j], 0).max() if (lb == j).sum() > 1 else 0 for j in range(kk))
                if ext < 0.072:
                    break
            C.extend(list(cs))
        self.C = np.array(C)
        log("init cubes", len(self.C), np.round(self.C, 3).tolist())

    def update(self, o):
        X = self.top_points(o)
        C = self.C.copy()
        at_home = np.linalg.norm(o.state[:3] - HOME) < 0.04
        if self.base_n is None or len(self.base_n) != len(C):
            self.base_n = np.full(len(C), 60.0)
        cnt = np.zeros(len(C))
        for it in range(3):
            if len(X) == 0:
                break
            d = np.linalg.norm(X[:, None] - C[None], axis=2)
            lab = d.argmin(1); dm = d.min(1)
            for k in range(len(C)):
                if k == self.held:
                    continue
                s = (lab == k) & (dm < ASSIGN_R)
                cnt[k] = s.sum()
                need = 8 if at_home else max(8, 0.6 * self.base_n[k])
                if s.sum() >= need:
                    C[k] = X[s].mean(0)
        if at_home:
            self.base_n = 0.8 * self.base_n + 0.2 * cnt
        mv = np.linalg.norm(C - self.C, axis=1)
        self.C = C
        self.hist.append((o.step_id, C.copy()))
        self.trk.write(json.dumps({"step": o.step_id, "C": np.round(C, 4).tolist(), "held": int(self.held)}) + "\n")
        return mv

    def moving(self, window=8, thr=0.0015):
        if len(self.hist) < window + 1:
            return True
        a = self.hist[-window - 1][1]; b = self.hist[-1][1]
        return np.linalg.norm(a - b, axis=1).max() > thr

    # ---------------- acting
    def step1(self, row):
        row = [float(np.clip(x, -1, 1)) for x in row]
        o = Obs(self.ctx.call("robomme_step", actions=[row]))
        self.obs = o
        if self.C is not None:
            self.update(o)
            still = self.prev_ee is not None and np.linalg.norm(o.state[:3] - self.prev_ee) < 0.002
            if np.linalg.norm(o.state[:3] - HOME) < 0.03 and still:
                self.note_motion()
            else:
                self.quiet_run = 0; self.move_run = 0; self.pre_quiet = 0
            self.prev_ee = o.state[:3].copy()
        if o.complete:
            log("TASK COMPLETE at", o.step_id)
        if o.ended:
            raise Abort("episode ended at %d complete=%s" % (o.step_id, o.complete))
        if time.time() - self.t0 > 1050:
            raise Abort("wall time")
        return o

    def hold(self, grip, n):
        for _ in range(n):
            self.step1([0, 0, 0, 0, 0, 0, grip])

    def servo(self, target, yaw=0.0, grip=-1.0, tol=0.003, ytol=0.03, max_iter=60, vmax=1.0):
        target = np.asarray(target, float)
        for it in range(max_iter):
            o = self.obs
            err = target - o.state[:3]
            Rc = aa_to_R(o.state[3:6])
            c, s = np.cos(yaw), np.sin(yaw)
            Rt = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])
            rot = R_to_aa(Rt @ Rc.T)
            if np.linalg.norm(err) < tol and np.linalg.norm(rot) < ytol:
                return True
            dv = np.clip(err / 0.05 * POS_GAIN, -vmax, vmax)
            dr = np.clip(rot / 0.5 * ROT_GAIN, -1, 1)
            self.step1([dv[0], dv[1], dv[2], dr[0], dr[1], dr[2], grip])
        return False

    def wait_settle(self, grip, min_steps=10, max_steps=120, quiet=12):
        n = 0
        while n < max_steps:
            self.hold(grip, 1); n += 1
            if n >= min_steps and not self.moving(window=quiet):
                break
        log("settled after", n, "step", self.obs.step_id)

    # ---------------- task phases
    def find_button(self, o):
        best = None
        for cam in ("wrist", "front"):
            if cam == "wrist" and o.pose["wrist"][0][2] < 0.3:
                continue
            P = o.cloud(cam)
            m = (P[..., 2] > 0.022) & (P[..., 2] < 0.06) & (P[..., 0] > -0.45) & (P[..., 0] < 0.25)
            X = P[m]
            if len(X) < 20:
                continue
            g = np.floor(X[:, :2] / 0.005).astype(int); g -= g.min(0)
            img = np.zeros(g.max(0) + 3, bool); img[g[:, 0] + 1, g[:, 1] + 1] = True
            lab, n = ndimage.label(img)
            L = lab[g[:, 0] + 1, g[:, 1] + 1]
            for i in range(1, n + 1):
                pts = X[L == i]
                if len(pts) < 15:
                    continue
                top = np.percentile(pts[:, 2], 95)
                if top < 0.0385:
                    sel = pts[pts[:, 2] > top - 0.004]
                    cand = (len(pts), sel[:, :2].mean(0), float(top), cam)
                    if best is None or cand[0] > best[0]:
                        best = cand
            if best is not None:
                break
        log("button", best and (best[1].round(3).tolist(), round(best[2], 3), best[3]))
        return None if best is None else best[1]

    def press(self, bxy):
        self.servo([bxy[0], bxy[1], 0.10], grip=1.0, tol=0.006)
        self.servo([bxy[0], bxy[1], 0.05], grip=1.0, tol=0.004, max_iter=15)
        # descend until stalled
        last = 9; stall = 0
        for i in range(40):
            z = self.obs.state[2]
            self.step1([float(np.clip((bxy[0] - self.obs.state[0]) / 0.05, -1, 1)),
                        float(np.clip((bxy[1] - self.obs.state[1]) / 0.05, -1, 1)),
                        float(np.clip((0.012 - z) / 0.05, -0.3, 0.3)), 0, 0, 0, 1.0])
            z2 = self.obs.state[2]
            if z2 < 0.028 and abs(z2 - z) < 0.0008:
                stall += 1
                if stall >= 3:
                    break
        log("press bottom z %.4f step %d" % (self.obs.state[2], self.obs.step_id))
        self.servo([bxy[0], bxy[1], 0.10], grip=1.0, tol=0.01)

    def detect_highlight(self, max_wait=90):
        """Hold at home until white discs appear; return bool array."""
        best = np.zeros(len(self.C)); seen = 0
        for i in range(max_wait):
            W = self.white_points(self.obs)
            cnt = np.zeros(len(self.C))
            if len(W):
                d = np.linalg.norm(W[:, None] - self.C[None], axis=2)
                for k in range(len(self.C)):
                    cnt[k] = (d[:, k] < 0.045).sum()
            best = np.maximum(best, cnt)
            if cnt.max() > 40:
                seen += 1
                if seen >= 6:
                    break
            if self.servo(HOME, grip=-1.0, tol=0.008, max_iter=1):
                self.hold(-1.0, 1)
        hl = best > max(25, 0.25 * best.max())
        log("highlight counts", best.astype(int).tolist(), "->", np.nonzero(hl)[0].tolist(), "step", self.obs.step_id)
        return hl

    def free_spot(self, xy, others):
        cands = [np.asarray(xy)]
        for r in (0.06, 0.09, 0.12):
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                cands.append(np.asarray(xy) + r * np.array([np.cos(a), np.sin(a)]))
        for c in cands:
            if len(others) == 0 or np.linalg.norm(others - c, axis=1).min() > 0.06:
                if -0.33 < c[0] < 0.10 and -0.28 < c[1] < 0.28:
                    return c
        return np.asarray(xy)

    # ---------------- schedule of shuffles
    def note_motion(self):
        """Called after each step while the arm is still at HOME; detects shuffle starts."""
        if len(self.hist) < 2:
            return
        mv = np.linalg.norm(self.hist[-1][1] - self.hist[-2][1], axis=1).max()
        st = self.hist[-1][0]
        if mv > MOVE_THR:
            if self.move_run == 0:
                self.pre_quiet = self.quiet_run
            self.move_run += 1
            self.quiet_run = 0
            self.last_move = st
            if self.move_run == 3 and self.pre_quiet >= 3 and (not self.starts or st - self.starts[-1] > 45):
                self.starts.append(st - 3)
                self.fit_period()
                log("shuffle start ~", st - 3, "period", self.period)
        else:
            self.quiet_run += 1
            if self.quiet_run >= 2:
                self.move_run = 0

    def fit_period(self):
        S = np.array(self.starts, float)
        if len(S) < 2:
            return
        if self.period is None:
            d = S[1] - S[0]
            if 50 <= d <= 160:
                self.period = float(d)
            return
        n = np.round((S - S[0]) / self.period)
        if n[-1] <= 0:
            return
        # least-squares phase and period
        A = np.stack([np.ones_like(n), n], 1)
        (ph, per), *_ = np.linalg.lstsq(A, S, rcond=None)
        res = S - (ph + per * n)
        if abs(per - self.period) < 0.2 * self.period and np.abs(res).max() < 8:
            self.period = float(per); self.phase = float(ph)
        else:
            log("period fit rejected", round(per, 1), np.round(res, 1).tolist())

    def next_start(self, now):
        """Predicted start of the shuffle that is ongoing or next (None if unknown)."""
        if not self.starts:
            return None
        if self.period is None:
            return None
        t = self.phase if self.phase is not None else float(self.starts[-1])
        while t + SHUFFLE_LEN <= now:
            t += self.period
        return t

    def wait_window(self, need, grip, max_wait=320):
        """Hold at HOME until quiet and the predicted next shuffle leaves `need` (+margin) steps."""
        n = 0
        if self.period is None and not self.starts:
            max_wait = min(max_wait, 40)   # no shuffles ever seen: don't burn the episode waiting
        while n < max_wait:
            now = self.obs.step_id
            nxt = self.next_start(now)
            recent = (now - self.last_move) < 6
            if self.quiet_run >= 6 and not recent:
                if nxt is not None and nxt - now >= need + LAG_MARGIN:
                    log("window ok at", now, "next shuffle ~%.0f" % nxt, "period", self.period, "waited", n)
                    return True
                if (nxt is not None and self.period - SHUFFLE_LEN - 6 < need + LAG_MARGIN
                        and now - self.last_move <= 10 and nxt - now > 0.5 * need):
                    log("window (short period, best effort) at", now, "next ~%.0f" % nxt)
                    return True
                if nxt is None and n > 0 and self.starts and now - self.last_move <= 12 and now - self.starts[-1] < 90:
                    # unknown period: act right after a shuffle just ended
                    log("window (unknown period) at", now, "waited", n)
                    return True
            self.hold(grip, 1); n += 1
        log("wait_window timeout at", self.obs.step_id)
        return False

    def go_home(self, grip):
        self.servo([self.obs.state[0], self.obs.state[1], max(self.obs.state[2], 0.12)], grip=grip, tol=0.01, max_iter=8)
        self.servo(HOME, grip=grip, tol=0.012)

    def sustained_motion(self, n=4, thr=0.003):
        if len(self.hist) < n + 1:
            return False
        H = np.array([h[1] for h in self.hist[-(n + 1):]])
        d = np.linalg.norm(np.diff(H, axis=0), axis=2)  # n x K
        if self.held >= 0:
            d[:, self.held] = 0
        near = np.linalg.norm(self.C - self.obs.state[None, :2], axis=1) < 0.08
        d[:, near] = 0   # partly hidden/excluded by the gripper: centroid shifts are not motion
        return bool((d.min(0) > thr).any())

    def others_moving(self, k):
        if len(self.hist) < 2:
            return False
        mv = np.linalg.norm(self.hist[-1][1] - self.hist[-2][1], axis=1)
        mv[k] = 0
        return mv.max() > 2 * MOVE_THR

    def pick(self, k):
        xy = self.C[k].copy()
        t0 = self.obs.step_id
        log("pick target", k, xy.round(3), "step", t0)
        self.servo([xy[0], xy[1], HOVER_Z], yaw=0.0, tol=0.006, max_iter=25)
        if self.sustained_motion():
            log("sustained cube motion at hover; abort pick", self.obs.step_id)
            return False
        yaw = 0.0; top = 0.042
        o = self.obs
        P = o.cloud("wrist")
        m = (P[..., 2] > TOP_Z_MIN) & (P[..., 2] < TOP_Z_MAX) & (o.depth["wrist"] > 0)
        pts = P[m]
        pts = pts[np.linalg.norm(pts[:, :2] - self.C[k][None], axis=1) < 0.028]
        if len(pts) >= 30:
            xy2 = pts[:, :2].mean(0)
            # recentre on the wrist cluster around xy2
            pts = P[m]; pts = pts[np.linalg.norm(pts[:, :2] - xy2[None], axis=1) < 0.028]
            xy2 = pts[:, :2].mean(0)
            yaw = square_yaw(pts[:, :2]); top = float(np.median(pts[:, 2]))
            log("refine", xy2.round(4), "yaw %.1f top %.3f n %d" % (np.degrees(yaw), top, len(pts)))
            if np.linalg.norm(xy2 - xy) > 0.03:
                log("refine far from track; abort pick"); return False
            xy = xy2
        else:
            log("refine: few points", len(pts))
        gz = max(top - GRASP_DZ, 0.012)
        self.servo([xy[0], xy[1], gz + 0.012], yaw=yaw, tol=0.005, max_iter=15)
        self.servo([xy[0], xy[1], gz], yaw=yaw, tol=0.003, max_iter=8)
        self.held = k
        self.hold(1.0, 7)
        w = self.obs.state[6] + self.obs.state[7]
        log("grasp width %.4f at" % w, self.obs.state[:3].round(3), "step", self.obs.step_id)
        if w < 0.01:
            self.held = -1
            return False
        self.servo([xy[0], xy[1], 0.12], yaw=yaw, grip=1.0, tol=0.01, max_iter=10)
        self.pick_info = (xy, top, yaw)
        log("picked", k, "steps", self.obs.step_id - t0)
        return True

    def place(self):
        k = self.held
        xy, top, yaw = self.pick_info
        t0 = self.obs.step_id
        others = np.delete(self.C, k, axis=0)
        spot = self.free_spot(xy, others)
        log("place", k, "at", spot.round(3))
        self.servo([spot[0], spot[1], 0.10], yaw=yaw, grip=1.0, tol=0.006)
        self.servo([spot[0], spot[1], top - GRASP_DZ + 0.006], yaw=yaw, grip=1.0, max_iter=15)
        self.hold(-1.0, 5)
        self.held = -1
        self.C[k] = spot
        self.done[k] = True
        log("placed", k, "steps", self.obs.step_id - t0)

    def run(self):
        self.obs = observe(self.ctx)
        o = self.obs
        log(VERSION, "start step", o.step_id, o.meta.get("instruction"))
        self.base_rgb = o.rgb["front"].astype(int); self.base_depth = o.depth["front"].copy()
        bxy = self.find_button(o)
        self.bxy = bxy
        self.init_cubes(o)
        self.done = np.zeros(len(self.C), bool)
        self.hl = np.zeros(len(self.C), bool)
        # highlights are timed (~steps 10-50 in ep1/ep2): read them from HOME first
        self.servo(HOME, grip=-1.0, tol=0.012, max_iter=5)
        self.hl = self.detect_highlight(max_wait=70)
        self.done = np.zeros(len(self.C), bool)
        self.servo(HOME, grip=-1.0, tol=0.008, max_iter=30)
        # press the button in the first quiet window (before any pick)
        # stay home until two shuffles measured the period (or timeout)
        n = 0
        while self.period is None and n < 260 and self.obs.remaining > 400:
            self.hold(-1.0, 1); n += 1
        log("period", self.period, "starts", self.starts, "step", self.obs.step_id)
        if bxy is not None:
            self.wait_window(NEED_PRESS, -1.0)
            self.press(bxy)
            self.go_home(-1.0)
        self.pressed_first = True
        self.loop()

    def loop(self):
        while self.obs.remaining > 60:
            if self.held >= 0:
                self.wait_window(NEED_PLACE, 1.0)
                self.place()
                self.go_home(-1.0)
                continue
            if not ((self.hl & ~self.done).any() and self.fails < 6 and self.obs.remaining > 150):
                break
            self.wait_window(NEED_PICK, -1.0)
            cand = np.nonzero(self.hl & ~self.done)[0]
            k = cand[np.argmax(self.C[cand, 0])]
            ok = self.pick(k)
            self.go_home(1.0 if ok else -1.0)
            if not ok:
                self.fails += 1
        log("all picked; final press", self.obs.step_id)
        if self.bxy is not None:
            self.wait_window(NEED_PLACE, -1.0, max_wait=150)
            self.press(self.bxy)
        self.servo(HOME, grip=-1.0, tol=0.02, max_iter=15)
        self.hold(-1.0, 10)


def main(ctx):
    os.makedirs(OUT, exist_ok=True)
    c = Ctl(ctx)
    try:
        try:
            c.run()
        except Abort:
            raise
        except Exception:
            log("ERROR in run", traceback.format_exc())
            for attempt in range(3):
                if c.C is None or c.hl is None or not getattr(c, "pressed_first", False):
                    break
                try:
                    c.obs = observe(ctx)
                    if c.obs.ended:
                        break
                    c.go_home(1.0 if c.held >= 0 else -1.0)
                    c.loop()
                    break
                except Abort:
                    raise
                except Exception:
                    log("ERROR in recovery", attempt, traceback.format_exc())
    except Abort as e:
        log("abort:", e)
    finally:
        try:
            with open(os.path.join(OUT, "log.txt"), "a") as f:
                f.write("\n".join(__import__("lib").LOG) + "\n")
        except Exception:
            pass
