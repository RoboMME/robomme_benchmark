"""VideoRepick controller: track the picked object through the conditioning video,
then pick+place it twice and press the button."""
import os, sys, json, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from robot import Robot, EpisodeEnded
from geom import depth_to_world
import vtrack, skills

OUT = "/workspace/output"

class Log:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, "controller_log.txt"), "a")
        self.t0 = time.time()
    def __call__(self, *a):
        s = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.f.write(s + "\n"); self.f.flush()
        except Exception:
            pass

def target_world_from_track(robot, meta, content, uv, log):
    ims = robot.images(meta, content)
    K = meta["camera_intrinsics"]["front"]["matrix"]; pose = meta["camera_poses"]["front"]
    P = depth_to_world(ims[("front", "depth")], K, pose)
    mask = vtrack.object_mask(ims[("front", "rgb")], P[..., 2])
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs - uv[0], ys - uv[1])
    sel = d < 7
    if sel.sum() < 5:
        sel = d < 12
    if sel.sum() < 3:
        log("no mask pixels near track; using raw pixel depth")
        v, u = int(round(uv[1])), int(round(uv[0]))
        return P[v, u, :2].copy(), P, mask
    W = P[ys[sel], xs[sel]]
    return W[:, :2].mean(0), P, mask

def choose_free_spot(obj_xy_all, origin, avoid, log):
    best = None
    for r in (0.08, 0.10, 0.12, 0.14):
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p = origin + r * np.array([np.cos(ang), np.sin(ang)])
            if not (-0.30 <= p[0] <= 0.02 and -0.22 <= p[1] <= 0.22):
                continue
            dmin = min([np.linalg.norm(p - q) for q in avoid] + [1.0])
            score = dmin - 0.3 * r
            if best is None or score > best[0]:
                best = (score, p, dmin)
    if best is None:
        return origin.copy()
    log(f"free spot {best[1].round(3)} clearance {best[2]:.3f}")
    return best[1]

def main(ctx):
    log = Log()
    robot = Robot(ctx, log=log)
    meta, content = robot.observe()
    try:
        import skills as _sk
        _sk.capture_finger_mask(robot, meta, content)
        fm = getattr(robot, "finger_mask", None)
        log("finger mask px", None if fm is None else int(fm.sum()))
    except Exception as e:
        log("finger mask error", repr(e))
    log("instruction:", meta.get("instruction"), "state", np.round(robot.state, 3).tolist(),
        "remaining", robot.remaining)
    # ---------- perception of the video ----------
    seq = meta.get("frame_sequence") or {}
    mp = os.path.join("/workspace", seq.get("manifest_path", ""))
    wxy = None; winfo = None
    try:
        import wtrack
        wpos, winfo = wtrack.run(mp, log=log)
        if wpos is not None:
            wxy = np.array(winfo["final_xy"])
            log("world-tracker", json.dumps(winfo))
    except Exception as e:
        log("world tracker failed:", repr(e), traceback.format_exc())
    target_uv = None; info = {}; k = 0; pos = None
    try:
        pos, cnts, info, _, _ = vtrack.run(mp, log=log)
        k = info["target_empty"] if info.get("t_pick") is not None else info["first_mover"]
        target_uv = pos[-1][k]
        log("legacy track info", json.dumps({kk: info[kk] for kk in ("target_empty", "first_mover", "t_pick", "t_back", "min_counts")}))
    except Exception as e:
        log("legacy tracking failed:", repr(e), traceback.format_exc())
    meta, content = robot.observe()
    P = None; lxy = None
    try:
        if target_uv is not None:
            lxy, P, mask = target_world_from_track(robot, meta, content, target_uv, log)
            ls = info.get("last_seen", {}).get(k)
            fc = info.get("final_counts", [99] * (k + 1))[k]
            if ls is not None and (fc < 8 or ls[0] < len(pos) - 1):
                log(f"legacy target occluded at end (count {fc}); last-seen xy {np.round(ls[1], 3)}")
                lxy = np.array(ls[1])
    except Exception as e:
        log("legacy target world failed", repr(e))
    if wxy is not None and winfo.get("final_count", 0) >= 8:
        txy = wxy; src = "world"
    elif lxy is not None:
        txy = lxy; src = "legacy"
    elif wxy is not None:
        txy = wxy; src = "world-weak"
    else:
        txy = np.array([-0.1, 0.0]); src = "default"
    log(f"TARGET xy {np.round(txy, 3).tolist()} from {src}; world {None if wxy is None else np.round(wxy, 3).tolist()} legacy {None if lxy is None else np.round(lxy, 3).tolist()}")
    try:
        json.dump(dict(target_xy=list(map(float, txy)), src=src, winfo=winfo), open(os.path.join(OUT, "track.json"), "w"), default=str)
    except Exception:
        pass
    # other objects (for free-spot selection) from front + wrist (home pose, calibrated)
    avoid = []
    try:
        ims = robot.images(meta, content)
        for cam in ("front", "wrist"):
            Pc = depth_to_world(ims[(cam, "depth")], meta["camera_intrinsics"][cam]["matrix"], meta["camera_poses"][cam])
            comps = vtrack.components(vtrack.object_mask(ims[(cam, "rgb")], Pc[..., 2]), min_px=6)
            for c in comps:
                W = Pc[c["ys"], c["xs"]]
                if W[:, 0].mean() > -0.45:
                    avoid.append(W[:, :2].mean(0))
    except Exception as e:
        log("avoid list failed", repr(e))
    try:
        b = skills.find_button(robot, "wrist", log=log) or skills.find_button(robot, "front", log=log)
        if b is not None:
            avoid.append(b["c"]); robot.button_hint = b
    except Exception as e:
        log("button pre-scan failed", repr(e))

    cur = np.array(txy)
    done_cycles = 0
    try:
        for cycle in range(2):
            held = False
            for attempt in range(3):
                held, cur2 = skills.pick(robot, cur, log=log)
                if held:
                    cur = cur2; break
                log(f"pick attempt {attempt} failed; retry")
                robot.set_grip(-1, 6)
                if cur2 is not None:
                    cur = np.asarray(cur2)
            if not held:
                log("could not pick; continuing anyway")
            av = [a for a in avoid if np.linalg.norm(a - cur) > 0.03]
            spot = choose_free_spot(None, cur, av, log)
            skills.place(robot, spot, log=log)
            # update avoid list: target now at spot
            cur = spot
            done_cycles += 1
            log(f"cycle {cycle} done, step {robot.step_id} remaining {robot.remaining}")
        skills.press_button(robot, log=log)
        if not robot.complete and not robot.ended:
            log("not complete after press; pressing again")
            skills.press_button(robot, log=log)
    except EpisodeEnded:
        log("episode ended; complete =", robot.complete)
    except Exception as e:
        log("controller error", repr(e), traceback.format_exc())
    log("final: complete", robot.complete, "ended", robot.ended, "step", robot.step_id)
