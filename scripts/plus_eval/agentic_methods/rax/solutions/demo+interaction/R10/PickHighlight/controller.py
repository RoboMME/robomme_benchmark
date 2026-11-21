"""PickHighlight controller: watch highlight rings at start, press button, pick+replace each highlighted cube, press button."""
import json, os, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot, log, LOG
from percep import backproject, white_mask, components
from tracker import Tracker, front_points, detect_front_cubes
from tracker2 import Tracker2, front_components
import v4
from v4 import Scene, hover_until, servo_xy, wrist_cube_near, FRONT_XBIAS

OUT = "/workspace/output"
TABLE_Z_MAX = 0.012
CUBE_ZLO, CUBE_ZHI = 0.034, 0.07


def save(name, obj):
    try:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    except Exception:
        pass


def save_img(name, arr):
    try:
        from PIL import Image
        Image.fromarray(arr).save(os.path.join(OUT, name))
    except Exception:
        pass


def detect_button(rgb, P):
    """Button = white region at low height, containing a grey raised disc."""
    wm = white_mask(rgb) & (P[..., 2] < 0.045) & (P[..., 2] > -0.01)
    best = None
    for c in components(wm, 15):
        pts = P[c[:, 0], c[:, 1]]
        ext = pts[:, :2].max(0) - pts[:, :2].min(0)
        if ext.max() > 0.12:
            continue
        score = len(c)
        if best is None or score > best[0]:
            best = (score, pts)
    if best is None:
        return None
    pts = best[1]
    xy = (pts[:, :2].min(0) + pts[:, :2].max(0)) / 2.0
    return np.array([xy[0], xy[1], float(np.percentile(pts[:, 2], 90))])


def detect_cubes(rgb, P, button_xy=None, exclude_r=0.045):
    """Raised objects (cube tops/sides) clustered; split oversized clusters."""
    m = (P[..., 2] > CUBE_ZLO) & (P[..., 2] < CUBE_ZHI)
    # exclude near-white (button) pixels
    cubes = []
    for c in components(m, 12):
        pts = P[c[:, 0], c[:, 1]]
        top = pts[pts[:, 2] > np.percentile(pts[:, 2], 40)]
        xy = top[:, :2].mean(0)
        if button_xy is not None and np.linalg.norm(xy - button_xy[:2]) < exclude_r:
            # the button disc: skip if mostly grey and low
            if np.percentile(pts[:, 2], 95) < 0.036:
                continue
        ext = top[:, :2].max(0) - top[:, :2].min(0)
        k = 1
        if len(c) < 25:
            continue
        if ext.max() > 0.075:
            k = int(min(4, np.ceil(ext.max() / 0.055)))
        if k == 1:
            cubes.append({"xy": xy, "n": len(c), "z": float(np.percentile(pts[:, 2], 95))})
        else:
            # simple k-means on xy of top points
            X = top[:, :2]
            cent = X[np.linspace(0, len(X) - 1, k).astype(int)]
            for _ in range(15):
                lab = np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), 1)
                cent = np.array([X[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
            for j in range(k):
                cubes.append({"xy": cent[j], "n": int((lab == j).sum()), "z": 0.04})
    return cubes


def ring_counts(cubes, P, new_white):
    pts = P[new_white]
    pts = pts[np.abs(pts[:, 2]) < 0.03]
    counts = np.zeros(len(cubes))
    if len(cubes) == 0 or len(pts) == 0:
        return counts
    C = np.array([c["xy"] for c in cubes])
    d = np.linalg.norm(pts[:, None, :2] - C[None], axis=-1)
    nearest = d.argmin(1)
    dmin = d.min(1)
    for i, j in enumerate(nearest):
        if dmin[i] < 0.05:
            counts[j] += 1
    return counts


def refine_cube(R, xy_guess, radius=0.03):
    """From the current wrist observation, find the raised component whose centroid is nearest the guess."""
    R.observe()
    P = R.world("wrist")
    rgb = R.imgs[("wrist", "rgb")]
    m = (P[..., 2] > CUBE_ZLO) & (P[..., 2] < CUBE_ZHI)
    best = None
    for c in components(m, 30):
        pts = P[c[:, 0], c[:, 1]]
        top = pts[pts[:, 2] > np.percentile(pts[:, 2], 50)]
        xy = top[:, :2].mean(0)
        d = float(np.linalg.norm(xy - xy_guess))
        ext = (top[:, :2].max(0) - top[:, :2].min(0)).max()
        if ext > 0.075:
            # merged cubes: take only points near the guess
            near = top[np.linalg.norm(top[:, :2] - xy_guess[None], axis=-1) < 0.03]
            if len(near) < 20:
                continue
            xy = near[:, :2].mean(0); d = float(np.linalg.norm(xy - xy_guess))
        if best is None or d < best[0]:
            best = (d, xy)
    if best is None or best[0] > radius:
        return xy_guess, False
    return best[1], True


def cube_yaw_and_axis(R, xy):
    """Using current wrist observation: estimate cube yaw from top-face points; choose finger axis with best clearance."""
    P = R.world("wrist")
    raised = (P[..., 2] > 0.012) & (P[..., 2] < CUBE_ZHI)
    pts = P[raised][:, :3]
    if len(pts) == 0:
        return None, {}
    dxy = pts[:, :2] - xy[None]
    r = np.linalg.norm(dxy, axis=1)
    top = dxy[(r < 0.03) & (pts[:, 2] > CUBE_ZLO)]
    yaw = 0.0
    if len(top) > 30:
        best = None
        for a in np.arange(0, np.pi / 2, np.pi / 90):
            c, s_ = np.cos(a), np.sin(a)
            u = top @ np.array([c, s_]); v = top @ np.array([-s_, c])
            area = (np.percentile(u, 98) - np.percentile(u, 2)) * (np.percentile(v, 98) - np.percentile(v, 2))
            if best is None or area < best[0]:
                best = (area, a)
        yaw = best[1]
    others = dxy[r > 0.032]
    info = {}
    cands = []
    for ax in (yaw, yaw + np.pi / 2):
        a = np.array([np.cos(ax), np.sin(ax)]); b = np.array([-a[1], a[0]])
        if len(others):
            along = np.abs(others @ a); across = np.abs(others @ b)
            bad = int(((along > 0.025) & (along < 0.065) & (across < 0.022)).sum())
        else:
            bad = 0
        cands.append((bad, ax))
    cands.sort(key=lambda t: t[0])
    info = {"yaw": round(float(yaw), 3), "cands": [(b, round(float(a), 3)) for b, a in cands]}
    return cands[0][1], info


def refine_button(R, bxy):
    R.observe()
    P = R.world("wrist")
    rgb = R.imgs[("wrist", "rgb")].astype(int)
    grey = ((rgb.max(-1) - rgb.min(-1)) < 25) & (rgb.mean(-1) > 70) & (rgb.mean(-1) < 200)
    sel = grey & (P[..., 2] > 0.02) & (P[..., 2] < 0.045) & (np.linalg.norm(P[..., :2] - bxy[None, None, :2], axis=-1) < 0.05)
    if sel.sum() < 30:
        return bxy, False
    xy = P[sel][:, :2].mean(0)
    return np.array([xy[0], xy[1], bxy[2]]), True


def press_button(R, bxy, tag):
    R.set_grip(1.0, 5)
    R.move_to([bxy[0], bxy[1], 0.12], max_iter=10, tol=0.004)
    b2, ok = refine_button(R, bxy)
    log("button refine", ok, np.round(b2, 4).tolist())
    if ok and np.linalg.norm(b2[:2] - bxy[:2]) < 0.04:
        bxy = b2
    R.move_to([bxy[0], bxy[1], 0.10], max_iter=10, tol=0.004)
    R.move_to([bxy[0], bxy[1], 0.045], max_iter=10, tol=0.003)
    zs = []
    for i in range(8):
        R.step([[0, 0, -0.3, 0, 0, 0, 1]] * 2)
        zs.append(round(float(R.state()[2]), 4))
        if R.ended:
            break
    log("press", tag, "z trace", zs)
    R.wait(3)
    R.move_to([bxy[0], bxy[1], 0.12], max_iter=6, tol=0.01)


LAST_PLACE = [None]


def pick_cube(R, xy, idx):
    xy = np.asarray(xy, float)
    for attempt in range(3):
        if R.ended:
            return False
        R.set_grip(-1.0, 4)
        R.move_to([xy[0], xy[1], 0.13], max_iter=10, tol=0.005)
        xy2, ok = refine_cube(R, xy, 0.03)
        ok1 = ok
        log("cube", idx, "attempt", attempt, "refine1", ok, np.round(xy2, 4).tolist())
        xy = xy2
        R.move_to([xy[0], xy[1], 0.075], max_iter=10, tol=0.003)
        xy2, ok = refine_cube(R, xy, 0.02)
        log("cube", idx, "refine2", ok, np.round(xy2, 4).tolist())
        if ok:
            xy = xy2
        elif not ok1:
            log("cube", idx, "not found near guess; skipping descent")
            R.wait(2)
            continue
        ax, info = cube_yaw_and_axis(R, xy)
        log("cube", idx, "axis", info)
        if ax is not None:
            R.rotate_fingers_to(ax)
            xy3, ok3 = refine_cube(R, xy, 0.02)
            if ok3:
                xy = xy3
            log("cube", idx, "after rotate", round(R.finger_angle(), 3), np.round(xy, 4).tolist())
        R.move_to([xy[0], xy[1], 0.075], max_iter=6, tol=0.003)
        R.move_to([xy[0], xy[1], 0.015], max_iter=10, tol=0.003)
        R.set_grip(1.0, 10)
        R.move_to([xy[0], xy[1], 0.12], max_iter=8, tol=0.008)
        R.wait(3)
        w = float(R.state()[6] + R.state()[7])
        log("cube", idx, "lifted z", round(float(R.state()[2]), 3), "width", round(w, 4))
        held = w > 0.012
        # put it back down where it was
        if held:
            LAST_PLACE[0] = np.array(xy[:2], float)
            R.move_to([xy[0], xy[1], 0.035], max_iter=8, tol=0.005)
            R.set_grip(-1.0, 8)
            R.move_to([xy[0], xy[1], 0.12], max_iter=6, tol=0.01)
            return True
        R.set_grip(-1.0, 4)
    return False


MOTION = []


def make_hook(T):
    def hook(R):
        if ("front", "depth") in R.imgs:
            MOTION.append(T.update(front_points(R)))
    return hook


def wait_quiet(R, min_steps, quiet_n=20, thr=0.0012, cap=200, z_hold=None):
    start = len(MOTION)
    for i in range(cap):
        if R.ended:
            return
        st = R.state()
        dz = 0.0
        if z_hold is not None and st[2] < z_hold - 0.01:
            dz = float(np.clip((z_hold - st[2]) / 0.05, -0.4, 0.4))
        R.step([[0, 0, dz, 0, 0, 0, R.grip]])
        seg = MOTION[start:]
        if len(seg) >= min_steps and len(seg) >= quiet_n and max(seg[-quiet_n:]) < thr:
            break
    seg = MOTION[start:]
    log("wait_quiet steps", len(seg), "max motion", round(max(seg) if seg else 0, 4), "step", R.last.get("step_id"))


def pick_tracked(R, S, j, idx):
    T = S.T
    def tgt():
        return T.pos[j] + np.array([FRONT_XBIAS, 0.0])
    for attempt in range(3):
        if R.ended:
            return False
        R.set_grip(-1.0, 3)
        # hover high, following the tracked cube
        hover_until(R, S, tgt, 0.12, need=0, cap=150, tol=0.008)
        # yaw estimate from wrist view around tracked target
        xyw = wrist_cube_near(R, tgt(), 0.035)
        ax = None
        if xyw is not None:
            ax, info = cube_yaw_and_axis(R, xyw)
            log("cube", idx, "yaw info", info)
        if ax is not None:
            R.rotate_fingers_to(ax, max_iter=4)
        # wait for a safe window, then go down quickly
        ok = hover_until(R, S, tgt, 0.075, need=45, cap=320, tol=0.005)
        if not ok:
            log("cube", idx, "no safe window; trying anyway")
        xy = tgt().copy()
        xyw = wrist_cube_near(R, xy, 0.028)
        log("cube", idx, "attempt", attempt, "track", np.round(xy, 4).tolist(), "wrist", None if xyw is None else np.round(xyw, 4).tolist(), "step", R.last.get("step_id"))
        if xyw is None:
            # target not confirmed by the wrist: never grasp blind; let tracker re-acquire
            log("cube", idx, "no wrist confirmation; retry")
            R.wait(5, -1.0)
            continue
        xy = xyw
        R.move_to([xy[0], xy[1], 0.075], max_iter=3, tol=0.003, vmax=0.03)
        R.move_to([xy[0], xy[1], 0.015], max_iter=6, tol=0.004, vmax=0.03)
        R.set_grip(1.0, 7)
        R.move_to([xy[0], xy[1], 0.10], max_iter=4, tol=0.01, vmax=0.03)
        w = float(R.state()[6] + R.state()[7])
        log("cube", idx, "lifted z", round(float(R.state()[2]), 3), "width", round(w, 4), "step", R.last.get("step_id"), "ended", R.ended)
        if R.ended:
            return w > 0.012
        held = w > 0.012
        if held:
            R.move_to([xy[0], xy[1], 0.035], max_iter=4, tol=0.006, vmax=0.03)
            R.set_grip(-1.0, 6)
            R.move_to([xy[0], xy[1], 0.12], max_iter=4, tol=0.01, vmax=0.03)
            T.pos[j] = xy - np.array([FRONT_XBIAS, 0.0]); T.vel[j] = 0
            return True
        R.set_grip(-1.0, 3)
        R.move_to([xy[0], xy[1], 0.12], max_iter=4, tol=0.01, vmax=0.03)
    return False


def main(ctx):
    t0 = time.time()
    R = Robot(ctx)
    summary = {"stage": "start", "version": "v6"}
    try:
        R.observe()
        home = R.state()[:3].copy()
        base_rgb = R.imgs[("wrist", "rgb")].copy()
        base_P = R.world("wrist")
        save_img("base_wrist.png", base_rgb)
        base_white = white_mask(base_rgb)
        init_front = [c.mean(0) for c in front_components(R.world("front"))]
        T = Tracker2(np.array(init_front))
        S = Scene(R, T)
        R.on_step = S.hook; R.single = True
        hl_P, nw, best = None, None, 0
        for k in range(12):
            R.wait(3 if k else 9, -1.0)
            R.observe()
            rgb = R.imgs[("wrist", "rgb")]
            P = R.world("wrist")
            new = white_mask(rgb) & ~base_white & (np.abs(P[..., 2]) < 0.03)
            log("ring poll step", R.last["step_id"], "new white", int(new.sum()))
            if new.sum() > best:
                best = int(new.sum()); hl_P, nw = P, new
                save_img("hl_wrist.png", rgb)
            if best > 150 and k >= 2:
                break
        button = detect_button(base_rgb, base_P)
        log("button", None if button is None else np.round(button, 4).tolist())
        fxy = T.pos.copy()
        counts = np.zeros(len(fxy))
        if nw is not None and len(fxy):
            rp = hl_P[nw][:, :2]
            d = np.linalg.norm(rp[:, None] - fxy[None], axis=-1)
            nn = d.argmin(1); ok_ = d.min(1) < 0.05
            for j in nn[ok_]:
                counts[j] += 1
        mx = counts.max() if len(counts) else 0
        hl_idx = [j for j in range(len(fxy)) if counts[j] >= max(12, 0.35 * mx)]
        log("front cubes", [(np.round(fxy[j], 3).tolist(), int(counts[j])) for j in range(len(fxy))])
        log("highlighted idx", hl_idx)
        summary.update(front_cubes=fxy, counts=counts, button=button, hl_idx=hl_idx)
        save("perception.json", summary)
        if button is None:
            button = np.array([home[0], home[1], 0.02])
        summary["stage"] = "press1"
        press_button(R, button, 1)
        R.move_to([R.state()[0], R.state()[1], 0.20], max_iter=3, tol=0.02)
        done = []
        results = []
        succ, fails = [], {}
        for rnd in range(3):
            while not R.ended and R.remaining() >= 120:
                st = R.state()
                todo = [j for j in hl_idx if j not in succ and fails.get(j, 0) <= rnd]
                if not todo:
                    break
                j = min(todo, key=lambda k: np.linalg.norm(T.pos[k] - st[:2]))
                summary["stage"] = "pick%d" % len(succ)
                ok = pick_tracked(R, S, j, len(succ))
                results.append((int(j), ok, rnd))
                if ok:
                    succ.append(j)
                else:
                    fails[j] = fails.get(j, 0) + 1
                log("pick", j, "ok", ok, "round", rnd, "step", R.last.get("step_id"), "t", round(time.time() - t0, 1), "reacq", T.reacq, "positions", np.round(T.pos, 3).tolist())
            if R.ended or R.remaining() < 120:
                break
            if all(j in succ for j in hl_idx) or rnd == 2:
                summary["stage"] = "press2"
                press_button(R, button, 2)
                R.wait(10)
                if R.ended:
                    break
                log("press 2 did not end episode; retry round")
        summary["picks"] = results
        if not R.ended:
            summary["stage"] = "press_final"
            R.on_step = None
            press_button(R, button, 3)
            R.wait(10)
        summary["stage"] = "done"
    except Exception as e:
        log("ERROR", repr(e), traceback.format_exc())
        summary["error"] = repr(e)
    summary["ended"] = R.ended
    summary["complete"] = R.complete
    summary["elapsed"] = time.time() - t0
    summary["log"] = LOG
    summary["starts"] = None
    save("summary.json", summary)
    log("FINAL", summary["stage"], "ended", R.ended, "complete", R.complete, "elapsed", round(time.time() - t0, 1))
