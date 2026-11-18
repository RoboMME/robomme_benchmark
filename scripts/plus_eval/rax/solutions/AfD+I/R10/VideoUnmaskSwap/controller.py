"""VideoUnmaskSwap controller: track containers through the conditioning video and the
first execution steps with front RGB-D, then lift the containers in instruction order."""
import os, sys, re, json, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tracker import load_frames, precompute, solve_video, Tracker, points_from_obs, make_mask_fn
from motion import Robot
from grasp import pick_and_return

OUT = Path("/workspace/output")
COLOR_WORDS = ["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink", "cyan", "gray"]

def log(*a):
    print(*a, flush=True)

def parse_instruction(text):
    """-> list of (color, all_flag) in order."""
    t = text.lower()
    t = t.split("then", 1)[1] if "then" in t else t
    out = []
    for m in re.finditer(r"\b(" + "|".join(COLOR_WORDS) + r")\b", t):
        pre = t[max(0, m.start() - 40):m.start()]
        # clause start = last comma / 'followed by' / 'finally'
        cut = max(pre.rfind(","), pre.rfind("followed"), pre.rfind("finally"), pre.rfind("then"))
        clause = pre[cut + 1:] if cut >= 0 else pre
        out.append((m.group(1), bool(re.search(r"\ball\b", clause))))
    return out

def wait_and_track(rb, tr, mask_fn, max_steps=260, still_need=15, log=log):
    still = 0
    for i in range(max_steps):
        if rb.ended or rb.remaining < 400:
            break
        rb.step([[0, 0, 0, 0, 0, 0, -1]])
        pts = points_from_obs(rb.meta, rb.imgs)
        moved, cnt = tr.update(pts, mask_fn)
        mx = float(moved.max()) if len(moved) else 0.0
        still = still + 1 if mx < 0.0008 else 0
        if i % 10 == 0:
            log(f"exec-track step={rb.meta['step_id']} maxmove={mx:.4f} still={still} cnt={cnt.tolist()} lost={tr.lost.tolist()}")
        if still >= still_need:
            break
    log(f"exec-track done at step {rb.meta['step_id']}")

def plan_targets(instr, colors, positions, start_xy, log=log):
    spec = parse_instruction(instr)
    log(f"spec={spec} colors={colors}")
    used = set()
    plan = []
    for color, all_flag in spec:
        idx = [k for k, c in enumerate(colors) if c == color and k not in used]
        if not idx:
            log(f"WARNING: no container for {color}")
            continue
        if not all_flag:
            idx = idx[:1] if len(idx) == 1 else idx
            # choose one: if several, prefer closest to the start
        cur = np.asarray(start_xy, float)
        chosen = []
        rem = list(idx)
        while rem:
            j = min(rem, key=lambda k: np.linalg.norm(positions[k] - cur))
            chosen.append(j); rem.remove(j); cur = positions[j]
            if not all_flag:
                break
        for j in chosen:
            used.add(j); plan.append(j)
    return plan

def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        log('CONTROLLER ERROR', repr(e)); traceback.print_exc()

def _main(ctx):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rb = Robot(ctx, log=log)
    meta, imgs = rb.observe()
    instr = meta.get("instruction", "")
    log("instruction:", instr, "step", meta.get("step_id"))
    seq = meta.get("frame_sequence") or {}
    mpath = Path("/workspace") / seq["manifest_path"]
    frames = load_frames(mpath)
    data = precompute(frames)
    vid = [d for d in data if d["phase"] == "video_demo"]
    R = solve_video(vid + [d for d in data if d["phase"] != "video_demo"], log=log)
    colors = R["colors"]
    mask_fn = make_mask_fn(R["mask"])
    log(f"video solved in {time.time()-t0:.1f}s; final={np.round(R['final'],3).tolist()}")
    tr = Tracker(R["final"])
    tr.lost = R["lost"].copy()
    wait_and_track(rb, tr, mask_fn)
    pos = tr.T.copy()
    log("positions:", [(c, np.round(p, 3).tolist()) for c, p in zip(colors, pos)])
    plan = plan_targets(instr, colors, pos, rb.state[:2])
    log("plan:", [(k, colors[k]) for k in plan])
    try:
        (OUT / "plan.json").write_text(json.dumps(dict(instr=instr, colors=colors, pos=pos.tolist(), plan=plan)))
    except Exception:
        pass
    for n, k in enumerate(plan):
        if rb.ended:
            break
        rb.move_to([rb.state[0], rb.state[1], 0.30], grip=-1.0, max_steps=40)
        rb.hold(1, -1.0)
        try:
            out = pick_and_return(rb, pos[k], log=log)
        except Exception as e:
            log("pick error", repr(e)); traceback.print_exc()
            continue
        log(f"picked #{n} k={k} {colors[k]} held={out['held']} step={rb.meta['step_id']} ended={rb.ended}")
    log(f"done ended={rb.ended} complete={rb.complete} step={rb.meta.get('step_id')} t={time.time()-t0:.1f}s")
