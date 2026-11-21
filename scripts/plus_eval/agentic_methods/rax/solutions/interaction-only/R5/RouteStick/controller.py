"""RouteStick controller: reproduce the conditioning-video stick path in the live layout.

1. Read the initial frame manifest (video frames carry robot state -> demo TCP path).
2. Detect white discs (front RGB-D) in the first video frame and the live frame.
3. Split the demo drawing path at disc visits; map each piece with a per-disc-pair
   similarity transform into the live layout.
4. Track the mapped path closed-loop at the demo drawing height.
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rslib, perceive, plan

OUT = Path("/workspace/output")

class Log:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.f = open(OUT / "controller_log.txt", "a")
        self.t0 = time.time()
    def __call__(self, *a):
        s = "[%.1fs] " % (time.time() - self.t0) + " ".join(str(x) for x in a)
        print(s); self.f.write(s + "\n"); self.f.flush()

def load_frame(frame):
    f = {(x["camera"], x["kind"]): x["path"] for x in frame["files"]}
    out = {}
    for cam in ("front", "wrist"):
        try:
            out[(cam, "rgb")] = np.array(Image.open(rslib.WS / f[(cam, "rgb")]).convert("RGB"))
            out[(cam, "depth")] = np.array(Image.open(rslib.WS / f[(cam, "depth")])).astype(float) * 0.001
        except Exception:
            pass
    return out

def build_plan(meta0, imgs0, log):
    man = json.loads((rslib.WS / meta0["frame_sequence"]["manifest_path"]).read_text())
    frames = sorted(man["frames"], key=lambda f: f["frame_index"])
    vid = [f for f in frames if f["phase"] == "video_demo"]
    log("video frames", len(vid))
    S = np.array([f["state"] for f in vid])
    De = perceive.detect_discs(meta0, imgs0)
    best = None
    for f in vid[:3] + vid[3:40:6]:
        try:
            D = perceive.detect_discs(f, load_frame(f))
        except Exception:
            continue
        if best is None or (len(best) != len(De) and len(D) > len(best)): best = D
        if len(D) == len(De): best = D; break
    if best is None or len(best) < 2 or len(De) < 2:
        log("WARN discs not detected (video %s live %d); identity mapping" % (None if best is None else len(best), len(De)))
        best = De if len(De) >= 2 else np.array([[0.0, -0.1], [0.0, 0.1]])
        if len(De) < 2: De = best.copy()
    if len(De) > len(best) >= 3:
        log("WARN extra live discs %d > %d; dropping off-line detections" % (len(De), len(best)))
        De = perceive.drop_outliers(De, len(best))
    if len(best) == len(De):
        Dv, De = perceive.consistent_order(best, De)
    else:
        log("WARN disc count mismatch video %d live %d -> matching" % (len(best), len(De)))
        Dv, De = perceive.match_mismatch(best, De)
    log("discs video", np.round(Dv, 3).tolist()); log("discs live", np.round(De, 3).tolist())
    zdraw = float(np.percentile(S[:, 2], 2))
    low = S[:, 2] < zdraw + 0.01
    idx = np.nonzero(low)[0]
    xy = S[idx[0]:idx[-1] + 1, :2]
    zs = S[idx[0]:idx[-1] + 1, 2]
    log("draw z", round(zdraw, 4), "draw frames", int(idx[0]), int(idx[-1]))
    visits = []
    if len(Dv) != len(De) or len(Dv) < 2:
        # fallback: rigid fit on centroid/principal direction of whatever discs we saw
        log("WARN disc count mismatch", len(Dv), len(De))
        _, cv, dv = perceive.order_along_line(Dv); _, ce, de = perceive.order_along_line(De)
        ang = np.arctan2(de[1], de[0]) - np.arctan2(dv[1], dv[0])
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        path = (xy - cv) @ R.T + ce
    else:
        visits = plan.segment_path(xy, Dv)
        log("visits", [(int(a), int(b)) for a, b in visits])
        path = plan.map_path(xy, Dv, De, visits)
        if len(visits) < 2:
            log("WARN few visits; rigid fallback")
            R, t = perceive.fit_rigid(Dv, De); path = xy @ R.T + t
        else:
            # include the approach before the first visit / after last unchanged in shape: skip
            pass
    # tool geometry: tip offset (tip - TCP) in video vs live, from wrist depth
    delta = np.zeros(3)
    try:
        tv, nv = perceive.stick_tip(vid[0], load_frame(vid[0]))
        tl, nl = perceive.stick_tip(meta0, imgs0)
        if tv is not None and tl is not None:
            ov = tv - np.array(vid[0]["state"][:3]); ol = tl - np.array(meta0["state"][:3])
            d = ol - ov
            log("tip offset video", np.round(ov, 4).tolist(), "live", np.round(ol, 4).tolist(), "delta", np.round(d, 4).tolist(), nv, nl)
            if np.linalg.norm(d) < 0.12: delta = d
            else: log("WARN implausible tool delta; ignoring")
    except Exception:
        log("tip offset failed", traceback.format_exc())
    # video landing direction (approach to first visit) for entry choice
    land_dir = None
    try:
        i0 = int(idx[0]); p_land = S[i0, :2]
        land_dir = p_land - xy[visits[0][0]] if len(Dv) == len(De) else None
    except Exception:
        pass
    return path, zdraw, Dv, De, delta, land_dir

def resample(path, discs, step=0.006, near_step=0.003, near_r=0.025):
    """Resample polyline with variable spacing (finer near discs)."""
    out = [path[0]]
    seg = np.diff(path, axis=0); L = np.linalg.norm(seg, axis=1)
    cum = np.r_[0, np.cumsum(L)]; total = cum[-1]
    s = 0.0
    while s < total:
        p = out[-1]
        dmin = np.min(np.linalg.norm(discs - p, axis=1)) if len(discs) else 1.0
        s += near_step if dmin < near_r else step
        s = min(s, total)
        i = min(np.searchsorted(cum, s) - 1, len(L) - 1); i = max(i, 0)
        f = (s - cum[i]) / max(L[i], 1e-9)
        out.append(path[i] + f * seg[i])
    return np.array(out)

def track(robot, path, z, log, discs=np.zeros((0, 2)), alpha0=0.4, catchup=0.008, max_steps=700, step=0.006):
    """Single-row closed loop: cmd = (target - pos)/alpha, alpha estimated online."""
    R = resample(path, discs, step=step)
    P = np.c_[R, np.full(len(R), z)]
    alpha = alpha0; k = 0; N = len(P); used = 0
    errs = []; vis_err = {}
    while k < N and not robot.ended and used < max_steps:
        cur = robot.pos
        e = P[k] - cur
        cmd = e / alpha / 0.05
        m = np.max(np.abs(cmd))
        if m > 0.8: cmd = cmd * 0.8 / m
        robot.step([list(cmd) + [0, 0, 0, 1.0]]); used += 1
        new = robot.pos
        ach = new - cur; c = cmd * 0.05
        cc = float(c @ c)
        if cc > 1e-8:
            a_obs = float(ach @ c) / cc
            alpha = float(np.clip(0.8 * alpha + 0.2 * a_obs, 0.25, 1.0))
        err = np.linalg.norm(P[k] - new); errs.append(err)
        if len(discs):
            dd = np.linalg.norm(discs - new[:2], axis=1); j = int(np.argmin(dd))
            vis_err[j] = min(vis_err.get(j, 1.0), float(dd[j]))
        if err < catchup: k += 1
        if used % 50 == 0:
            log("track k %d/%d err %.4f alpha %.3f step %s" % (k, N, err, alpha, robot.meta.get("step_id")))
    log("track done k %d/%d used %d maxerr %.4f meanerr %.4f alpha %.3f" % (k, N, used, max(errs or [0]), float(np.mean(errs or [0])), alpha))
    log("closest approach per disc", {int(a): round(b, 4) for a, b in sorted(vis_err.items())})
    return alpha

def choose_entry(start, De, blocks, prefer=None, dist=0.04, log=print):
    best = None
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        u = np.array([np.cos(ang), np.sin(ang)])
        seg = start + np.outer(np.linspace(0, 1, 9), u * dist)
        cb = np.min(np.linalg.norm(seg[:, None] - blocks[None], axis=2)) if len(blocks) else 0.1
        others = [d for d in De if np.linalg.norm(d - start) > 0.01]
        cd = np.min(np.linalg.norm(seg[1:, None] - np.array(others)[None], axis=2)) if others else 0.1
        score = min(cb, 0.05) + min(cd - 0.035, 0.0) * 2 + (0.01 * float(u @ prefer) if prefer is not None else 0)
        if best is None or score > best[0]: best = (score, u, cb, cd)
    log("entry dir", np.round(best[1], 3).tolist(), "block clr %.3f disc clr %.3f" % (best[2], best[3]))
    return start + best[1] * dist

def main(ctx):
    log = Log()
    robot = rslib.Robot(ctx, log)
    try:
        meta0, imgs0 = robot.observe()
        log("instruction", meta0.get("instruction"), "state", np.round(meta0["state"], 3).tolist(),
            "limits", meta0.get("episode_limits"))
        path, zdraw, Dv, De, delta, land_dir = build_plan(meta0, imgs0, log)
        np.save(OUT / "plan_path.npy", path)
        blocks = perceive.block_points(meta0, imgs0)
        # TCP targets = desired tip targets - live tool delta
        zt = zdraw - delta[2]; dxy = delta[:2]
        P = path - dxy
        start_disc = path[0]
        prefer = None
        if land_dir is not None and np.linalg.norm(land_dir) > 1e-3:
            prefer = land_dir / np.linalg.norm(land_dir)
        entry = choose_entry(start_disc, De, blocks, prefer, 0.04, log)
        log("tcp z target %.4f (draw z %.4f, delta %s)" % (zt, zdraw, np.round(delta, 4).tolist()))
        e, n = robot.move_to(np.r_[entry - dxy, zt + 0.07], max_steps=80, tol=0.005, vmax=0.8)
        log("above entry err %.4f steps %d step_id %s" % (e, n, robot.meta.get("step_id")))
        e, n = robot.move_to(np.r_[entry - dxy, zt], max_steps=40, tol=0.003, vmax=0.5)
        log("at entry err %.4f steps %d step_id %s" % (e, n, robot.meta.get("step_id")))
        registered = False
        for attempt in range(2):
            track(robot, np.vstack([entry, start_disc]) - dxy, zt, log, discs=np.zeros((0, 2)), step=0.003)
            robot.move_to(np.r_[P[0], zt], max_steps=10, tol=0.002, vmax=0.3)
            robot.step([[0, 0, 0, 0, 0, 0, 1.0]] * 2)
            robot.observe()
            registered = perceive.red_near(robot.meta, robot.imgs, start_disc)
            log("start registration attempt %d: red=%s step %s" % (attempt, registered, robot.meta.get("step_id")))
            if registered or robot.ended or attempt == 1: break
            # back out and re-enter
            track(robot, np.vstack([start_disc, entry]) - dxy, zt, log, discs=np.zeros((0, 2)))
        track(robot, P, zt, log, discs=De - dxy)
        e, n = robot.move_to(np.r_[P[-1], zt], max_steps=15, tol=0.002, vmax=0.3)
        log("final err %.4f ended %s complete %s step %s" % (e, robot.ended, robot.complete, robot.meta.get("step_id")))
        held = 0
        while not robot.ended and held < 60:
            robot.step([[0, 0, 0, 0, 0, 0, 1.0]] * 5); held += 5
        log("end: ended", robot.ended, "complete", robot.complete, "step", robot.meta.get("step_id"))
    except Exception:
        log("EXCEPTION", traceback.format_exc())
