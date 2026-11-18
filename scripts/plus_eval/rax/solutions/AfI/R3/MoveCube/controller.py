"""MoveCube controller: imitate the manner shown in the conditioning video."""
import os, sys, time, math, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from motion import Robot, Ended, yaw_of
from perception import parse_reply, scene, cube_estimate, target_estimate, obstacle_points, seg_dist
from manner import load_video_states, classify
import hook, pickpush, stick, demo

OUT = "/workspace/output"


class Log:
    def __init__(self):
        self.t0 = time.time()
        os.makedirs(OUT, exist_ok=True)
        self.f = open(os.path.join(OUT, "controller_log.txt"), "a")

    def __call__(self, *a):
        s = f"[{time.time()-self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.f.write(s + "\n"); self.f.flush()
        except Exception:
            pass


def perceive_start(rb, log):
    o = parse_reply(rb.observe())
    sc = scene(o)
    cube = cube_estimate(o)
    target = target_estimate(o)
    log(f"start: cube {None if cube is None else np.round(cube,3)} target {None if target is None else np.round(target,3)}")
    h = sc.get("hook")
    if h is not None:
        log(f"hook cam={h['cam']} handle {np.round(h['handle'],3)} head {np.round(h['head'],3)} len {h['length']:.3f} chir {h['chirality']}")
    excl = []
    if h is not None:
        pass
    if cube is not None:
        excl.append((cube, 0.035))
    if target is not None:
        excl.append((target, 0.04))
    obst = obstacle_points(o, exclude_xy=excl)
    # robot base region
    if len(obst):
        obst = obst[~((obst[:, 0] < -0.42) & (np.abs(obst[:, 1]) < 0.2))]
    log(f"obstacle points {len(obst)}")
    return o, sc, cube, target, obst


def inspect_hook(rb, h, log):
    """Hover the wrist camera over the hook for a close top-down view; return refined hook or None."""
    from perception import find_hook
    mid = (np.asarray(h["handle"][:2]) + np.asarray(h["head"][:2])) / 2
    cam_off = np.array([0.046, 0.02])  # wrist camera offset from TCP at yaw 0
    ee = mid - cam_off
    z = 0.30
    ok = rb.move_to([ee[0], ee[1], z], yaw=0.0, tol=0.01, ytol=0.05, max_steps=80, vmax=0.5, chunk=2)
    rb.hold(3)
    o = parse_reply(rb.observe())
    h2 = find_hook(o, cams=("wrist",))
    if h2 is None:
        log("close-up: hook not found in wrist view")
        return None, o
    log(f"close-up hook: handle {np.round(h2['handle'],3)} head {np.round(h2['head'],3)} len {h2['length']:.3f} "
        f"chir {h2['chirality']} pink_diag {h2['pink_diag']:.3f} (was handle {np.round(h['handle'],3)} head {np.round(h['head'],3)})")
    if np.linalg.norm(h2["handle"][:2] - h["handle"][:2]) > 0.06:
        log("close-up hook disagrees with overview; keeping overview")
        return None, o
    return h2, o


def tool_from_view(o, want_diag, want_len, log):
    tools = demo.tool_list(o)
    if not tools:
        return None, tools
    for t in tools:
        log(f"  tool: handle {np.round(t['handle'],3)} head {np.round(t['head'],3)} len {t['length']:.3f} head_diag {t['head_diag']:.3f} cam {t['cam']}")
    tools.sort(key=lambda t: abs(t["head_diag"] - want_diag) * 3 + abs(t["length"] - want_len))
    return tools[0], tools


def hook_dict(t):
    """Convert a tool record into the hook-routine format (s, c, chirality from pink points)."""
    pink = t["pink_P"]; handle = t["handle"]; head = t["head"]
    s = t["d"]; perp = np.array([-s[1], s[0]])
    side = np.mean((pink[:, :2] - handle[:2]) @ perp)
    c = perp if side > 0 else -perp
    return dict(cam=t["cam"], handle=handle, head=head, s=s, c=c, length=t["length"], chirality=1 if side > 0 else -1)


def run_tool(rb, o, sc, cube, target, obst, ana, log):
    want_diag = 0.09; want_len = 0.29
    if ana is not None and ana.get("tool"):
        want_diag = ana["tool"]["head_diag"]; want_len = ana["tool"]["length"]
    t, tools = tool_from_view(o, want_diag, want_len, log)
    if t is None:
        log("no tool in view; fallback push with gripper")
        return pickpush.run_push(rb, o, cube, target, obst, closed=True, log=log)
    # close-up of the chosen tool with the wrist camera
    mid = (np.asarray(t["handle"][:2]) + np.asarray(t["head"][:2])) / 2
    ee = mid - np.array([0.046, 0.02])
    rb.move_to([ee[0], ee[1], 0.30], yaw=0.0, tol=0.01, ytol=0.05, max_steps=80, vmax=0.5, chunk=2)
    rb.hold(3)
    o2 = parse_reply(rb.observe())
    t2 = None
    for c in demo.tool_list(o2, cams=("wrist",)):
        if np.linalg.norm(c["handle"][:2] - t["handle"][:2]) < 0.06 and abs(c["head_diag"] - t["head_diag"]) < 0.04:
            t2 = c
    if t2 is not None:
        log(f"close-up tool: handle {np.round(t2['handle'],3)} head {np.round(t2['head'],3)} len {t2['length']:.3f} diag {t2['head_diag']:.3f}")
        t = t2
    else:
        log("close-up did not confirm tool; using overview")
    if t["head_diag"] >= 0.06:
        sc = dict(sc); sc["hook"] = hook_dict(t)
        return hook.run_hook(rb, sc, cube, target, obst, log=log)
    rel = np.array([-0.037, 0.0]); phi = -math.pi / 2
    if ana is not None and ana.get("head_rel_u") is not None and ana.get("phi_rel") is not None:
        rel = np.asarray(ana["head_rel_u"], float); phi = float(ana["phi_rel"])
    return stick.run_stick(rb, t, cube, target, rel, phi, obst, log=log)


def main(ctx):
    log = Log()
    rb = Robot(ctx, log=log)
    try:
        r = rb.observe()
        meta = rb.meta
        log("instruction:", meta.get("instruction"), "remaining", rb.remaining)
        manner = "hook"
        ana = None
        try:
            frames = demo.load_frames(meta)
            if frames:
                ana = demo.analyse(frames, log=log)
                log("demo analysis:", {k: v for k, v in ana.items() if k not in ("tools0",)})
        except Exception:
            log("demo analysis error", traceback.format_exc())
        try:
            S, msg = load_video_states(meta)
            log("video:", msg)
            if S is not None:
                manner, info = classify(S)
                log("state classifier:", manner, info)
        except Exception as e:
            log("manner classification error", repr(e))
        if ana is not None and ana.get("kind") in ("tool", "pick", "push_closed", "push_open"):
            manner = ana["kind"]
        if os.environ.get("MOVECUBE_FORCE_MANNER"):  # offline tests only
            manner = os.environ["MOVECUBE_FORCE_MANNER"]
        log("MANNER:", manner)
        o, sc, cube, target, obst = perceive_start(rb, log)
        if cube is None or target is None:
            rb.move_to([-0.15, 0.0, 0.45], yaw=0.0, tol=0.01, max_steps=60)
            o, sc, cube, target, obst = perceive_start(rb, log)
        if cube is None or target is None:
            log("cube/target not found; stopping")
            return
        if manner in ("tool", "hook"):
            run_tool(rb, o, sc, cube, target, obst, ana, log)
        elif manner == "pick":
            pickpush.run_pick(rb, o, cube, target, obst, log=log)
        elif manner == "push_closed":
            pickpush.run_push(rb, o, cube, target, obst, closed=True, log=log)
        else:
            pickpush.run_push(rb, o, cube, target, obst, closed=False, log=log)
        log("done; steps used", rb.steps, "remaining", rb.remaining)
    except Ended:
        log("episode ended; steps", rb.steps)
    except Exception:
        log("ERROR", traceback.format_exc())
