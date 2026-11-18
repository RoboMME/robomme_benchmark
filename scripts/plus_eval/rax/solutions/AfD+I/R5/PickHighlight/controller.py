"""PickHighlight controller v3: highlight memory + per-step cube identity tracking
1) hold still from step 0, track every cube each step (front+wrist), accumulate highlight votes per track
2) wait until cube shuffling has stopped
3) press button  4) pick each highlighted track (re-acquire with wrist), put back  5) press button."""
import os, sys, json, time, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion import Robot
from percep import find_button, highlight_votes
from tracker import Tracker, detections
from skills import pick_cube, press_button

OUT = "/workspace/output"

def log_factory():
    os.makedirs(OUT, exist_ok=True)
    f = open(os.path.join(OUT, "controller_log.txt"), "a")
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        try:
            f.write(s + "\n"); f.flush()
        except Exception:
            pass
    return log

def main(ctx):
    log = log_factory()
    try:
        _main(ctx, log)
    except Exception as e:
        import traceback
        log("EXCEPTION", repr(e), traceback.format_exc())

def _main(ctx, log):
    t0 = time.time()
    rb = Robot(ctx, log=log)
    log("start step", rb.meta.get("step_id"), "state", np.round(rb.state, 3).tolist())
    btn = find_button(rb.meta, rb.imgs, "wrist") or find_button(rb.meta, rb.imgs, "front")
    bxy = None if btn is None else np.asarray(btn[1], float)
    tk = Tracker(log=log)
    tk.init(detections(rb.meta, rb.imgs), rb.meta["step_id"])
    log("init tracks", [(t["id"], np.round(t["xy"], 3).tolist()) for t in tk.tracks])
    state = {"phase": "watch"}

    def on_obs(r):
        st = r.meta["step_id"]
        dets = detections(r.meta, r.imgs, ("front", "wrist"))
        mv = tk.update(dets, st, allow_new=(state["phase"] == "watch"))
        if state["phase"] == "watch" and bxy is not None:
            vw = highlight_votes(r.meta, r.imgs, "wrist", [{"xyz": t["xy"]} for t in tk.tracks], bxy, r=0.04)
            vf = highlight_votes(r.meta, r.imgs, "front", [{"xyz": t["xy"]} for t in tk.tracks], bxy, r=0.04)
            for t, a, b in zip(tk.tracks, vw, vf):
                t["votes"] = max(t["votes"], a + 1.5 * b)
            state["last_votes"] = int(max(vw + vf) if (vw or vf) else 0)
        state["mv"] = mv

    rb.on_obs = on_obs
    rb.chunk = 1
    quiet = 0; seen_hl = False; hl_off_step = None
    while not rb.ended and rb.meta["step_id"] < 170:
        rb.hold(1, grip=-1.0)
        st = rb.meta["step_id"]
        lv = state.get("last_votes", 0)
        if lv > 50:
            seen_hl = True
        elif seen_hl and hl_off_step is None:
            hl_off_step = st
            log("highlight off at step", st)
        mv = state.get("mv", 0)
        quiet = quiet + 1 if mv < 0.004 else 0
        if st % 10 == 0:
            log("watch step", st, "votes", lv, "maxmove %.3f" % mv, "quiet", quiet, "ntracks", len(tk.tracks))
        if hl_off_step is not None and st > hl_off_step + 25 and quiet >= 20:
            break
    log("watch done step", rb.meta["step_id"], "t=%.1f" % (time.time() - t0))
    state["phase"] = "act"
    vot = np.array([t["votes"] for t in tk.tracks])
    mx = vot.max() if len(vot) else 0
    hl_ids = [t["id"] for t in tk.tracks if mx > 40 and t["votes"] > 0.4 * mx]
    log("tracks", [(t["id"], np.round(t["xy"], 3).tolist(), int(t["votes"])) for t in tk.tracks])
    log("highlighted ids", hl_ids)
    try:
        json.dump({"tracks": [{"id": t["id"], "xy": t["xy"].tolist(), "votes": float(t["votes"])} for t in tk.tracks],
                   "hl": hl_ids, "button": None if bxy is None else bxy.tolist(), "moves": tk.max_move_hist},
                  open(os.path.join(OUT, "scene.json"), "w"))
    except Exception as e:
        log("json err", e)
    if bxy is None:
        log("ABORT no button"); return
    rb.chunk = 2
    press_button(rb, bxy, log=log)
    log("press1 done step", rb.meta["step_id"], "ended", rb.ended)
    # let any post-press motion settle while tracking
    rb.chunk = 1
    q = 0
    for _ in range(40):
        if rb.ended: break
        rb.hold(1)
        q = q + 1 if state.get("mv", 0) < 0.004 else 0
        if q >= 10: break
    rb.chunk = 2
    todo = sorted(hl_ids, key=lambda j: -np.linalg.norm(tk.get(j)["xy"] - bxy[:2]))
    for tid in todo:
        if rb.ended:
            break
        if time.time() - t0 > 1050:
            log("time budget low; stop picking"); break
        held, info = pick_cube(rb, tk, tid, button_xy=bxy[:2], log=log)
        log("pick", tid, "held", held, "step", rb.meta["step_id"], "ended", rb.ended, "succ", rb.success, "t=%.0f" % (time.time() - t0))
        if not held and not rb.ended:
            log("retry pick", tid)
            held, info = pick_cube(rb, tk, tid, button_xy=bxy[:2], log=log)
            log("pick", tid, "held", held, "step", rb.meta["step_id"])
    if not rb.ended:
        press_button(rb, bxy, log=log)
        rb.hold(5)
        log("final press step", rb.meta["step_id"], "ended", rb.ended, "success", rb.success)
    log("done t=%.1f calls %d" % (time.time() - t0, rb.n_calls))
