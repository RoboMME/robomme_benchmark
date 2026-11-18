"""PatternLock controller v3.

Pattern = grid nodes where the conditioning-video tool dwelled while low (video TCP
states in the reset manifest).  Video grid -> execution grid via a similarity fit of
the detected 5x5 button grids (front depth).  Execution: demo-style approach (descend
beside the first node, slide in), straight node-to-node strokes at constant speed,
and a dwell on every node so it registers.
"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion import Robot, rotvec_to_R  # noqa
import perception as pc  # noqa

OUT = Path("/workspace/output")
SPEED = 0.022       # m per step on strokes (demo ~0.005-0.008); native horizon ~450 steps
SLIDE_SPEED = 0.006 # m per step for the first slide-in
DWELL = 6           # steps held on every node
HOVER = 0.04        # m above drawing height for the approach


def log(*a):
    print(*a, flush=True)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "controller_log.txt", "a") as f:
            f.write(" ".join(str(x) for x in a) + "\n")
    except Exception:
        pass


def load_manifest(pkt):
    seq = pkt.get("frame_sequence") or {}
    mp = seq.get("manifest_path")
    return json.loads((Path("/workspace") / mp).read_text())


def merge_red(vid, gv, nodes):
    """Cross-check dwell nodes with the red-highlight onsets; add red-only nodes by time."""
    try:
        red = pc.red_onsets(vid, gv["idx"])
    except Exception as e:
        log("red_onsets error", repr(e))
        return nodes
    log("red onsets", red)
    dw = [n["idx"] for n in nodes]
    if [k for k, _ in red] == dw:
        log("pattern check: dwell == red")
        return nodes
    log("pattern check: MISMATCH dwell", dw, "red", [k for k, _ in red])
    out = list(nodes)
    for k, pos in red:
        if k in dw:
            continue
        # insert before the first dwell node that started after this onset (red lights ~0.3 s into a dwell)
        i = 0
        while i < len(out) and out[i]["start"] <= pos - 8:
            i += 1
        out.insert(i, {"idx": k, "point": np.asarray(gv["idx"][k]), "start": pos - 8, "end": pos, "dist": 0.0})
        log("inserted red-only node", k, "at", i)
    return out


def plan(pkt):
    man = load_manifest(pkt)
    frames = man["frames"]
    vid = sorted([f for f in frames if f.get("phase") == "video_demo"], key=lambda f: f["frame_index"])
    ex = [f for f in frames if f.get("phase") == "execution"]
    ex = ex[-1] if ex else frames[-1]
    tv, te = pc.top_centers(vid[0]), pc.top_centers(ex)
    if len(tv) != 25 or len(te) != 25:
        log("WARN button count video", len(tv), "exec", len(te))
    gv, ge = pc.grid_index(tv), pc.grid_index(te)
    log("grid video n", len(gv["idx"]), "theta", round(np.degrees(gv["theta"]), 2), "s", round(gv["spacing"], 4),
        "| exec n", len(ge["idx"]), "theta", round(np.degrees(ge["theta"]), 2), "s", round(ge["spacing"], 4))
    keys = [k for k in gv["idx"] if k in ge["idx"]]
    A, t, res = pc.fit_similarity([gv["idx"][k] for k in keys], [ge["idx"][k] for k in keys])
    log("transform A", np.round(A, 4).tolist(), "t", np.round(t, 4).tolist(), "resid max", round(float(res.max()), 4))
    nodes, zmin = pc.dwell_nodes(vid, gv)
    log("video nodes", [(n["idx"], n["start"], n["end"]) for n in nodes])
    nodes = merge_red(vid, gv, nodes)
    st = np.array([f["state"][:3] for f in vid])
    # drawing height: demo z during dwells, corrected by button-top height change
    zd = [st[n["start"]:n["end"] + 1, 2] for n in nodes]
    z_draw = float(np.median(np.concatenate(zd))) if zd else zmin
    ztv = float(np.median([b[2] for b in tv]))
    zte = float(np.median([b[2] for b in te]))
    z_draw += (zte - ztv)
    # map node centres to execution grid
    ecent = {k: np.asarray(v) for k, v in ge["idx"].items()}
    targets = []
    for n in nodes:
        p = A @ np.asarray(gv["idx"][n["idx"]]) + t
        k = min(ecent, key=lambda kk: np.linalg.norm(ecent[kk] - p))
        d = float(np.linalg.norm(ecent[k] - p))
        if d > 0.5 * ge["spacing"]:
            log("WARN node map far", n["idx"], k, d)
        targets.append((k, ecent[k]))
    # demo approach point: first low frame of first dwell's approach
    low = np.nonzero(st[:, 2] < zmin + 0.012)[0]
    approach = None
    if len(low) and nodes:
        a = st[low[0], :2]
        off = A @ (a - np.asarray(gv["idx"][nodes[0]["idx"]]))
        if np.linalg.norm(off) > 0.03:
            off = off / np.linalg.norm(off) * 0.03
        if np.linalg.norm(off) < 0.008:
            off = np.array([-0.015, 0.0])
        approach = targets[0][1] + off
    log("exec targets", [(k, np.round(p, 4).tolist()) for k, p in targets], "z_draw", round(z_draw, 4),
        "approach", None if approach is None else np.round(approach, 4).tolist())
    return targets, z_draw, approach


def stroke(r, start, end, z, speed, tol=0.002, max_extra=15):
    """Constant-speed straight line from start to end at height z with feedback."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    L = float(np.linalg.norm(end - start))
    n = max(1, int(np.ceil(L / speed)))
    for i in range(1, n + 1):
        if r.ended or r.remaining() <= 0:
            return
        p = start + (end - start) * (i / n)
        r.step([r.row_towards([p[0], p[1], z], vmax=1.0)])
    e = 0
    while e < max_extra and not r.ended and r.remaining() > 0:
        if np.linalg.norm(r.state()[:2] - end) < tol and abs(r.state()[2] - z) < 0.003:
            break
        r.step([r.row_towards([end[0], end[1], z], vmax=0.3)])
        e += 1


def dwell(r, p, z, n):
    for _ in range(n):
        if r.ended or r.remaining() <= 0:
            return
        r.step([r.row_towards([p[0], p[1], z], vmax=0.2)])


def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        log("CONTROLLER ERROR", repr(e), traceback.format_exc())


def _main(ctx):
    t0 = time.time()
    r = Robot(ctx, log=log)
    pkt = r.observe()
    log("limits", pkt.get("episode_limits"))
    targets, z_draw, approach = plan(pkt)
    r.R_ref = rotvec_to_R(r.state()[3:6])
    if not targets:
        log("no targets; stop")
        return
    first = targets[0][1]
    ap = approach if approach is not None else first
    # hover above approach point, descend, slide in
    r.goto([ap[0], ap[1], z_draw + HOVER], tol=0.005, vmax=1.0, max_steps=60)
    r.goto([ap[0], ap[1], z_draw], tol=0.002, vmax=0.3, max_steps=30)
    log("descended at", np.round(r.state()[:3], 4), "step", r.last.get("step_id"))
    stroke(r, r.state()[:2], first, z_draw, SLIDE_SPEED)
    dwell(r, first, z_draw, DWELL)
    log("node", targets[0][0], "at", np.round(r.state()[:3], 4), "step", r.last.get("step_id"))
    for k, p in targets[1:]:
        if r.ended:
            break
        stroke(r, r.state()[:2], p, z_draw, SPEED)
        dwell(r, p, z_draw, DWELL)
        log("node", k, "at", np.round(r.state()[:3], 4), "step", r.last.get("step_id"),
            "ended", r.ended, "complete", r.task_complete)
    # hold on the last node waiting for the verdict
    n = 0
    while not r.ended and n < 20 and r.remaining() > 0:
        dwell(r, targets[-1][1], z_draw, 1)
        n += 1
    log("final step", r.last.get("step_id"), "ended", r.ended, "complete", r.task_complete,
        "wall", round(time.time() - t0, 1))
