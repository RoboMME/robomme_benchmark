import os, re, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lib import *

NUM = {"one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
SWING_Z = 0.11

def parse_instruction(ins):
    s = ins.lower()
    m = re.search(r"pick up the (\w+) cube", s)
    color = m.group(1) if m else "red"
    if color not in ("red", "green", "blue"):
        color = "red"
    n = None
    m = re.search(r"(\w+) times", s)
    if m:
        w = m.group(1)
        n = int(w) if w.isdigit() else NUM.get(w)
    if n is None:
        for w, v in NUM.items():
            if re.search(r"\b%s\b" % w, s):
                n = v; break
    if n is None:
        n = 1
    ir, il = s.find("right-side"), s.find("left-side")
    first = "right" if (ir >= 0 and (il < 0 or ir < il)) else "left"
    return color, n, first

def find_scene(o, cam):
    """targets (list of xy), button xyz from white components"""
    rgb = o[(cam, "rgb")]; P = world_map(o, cam)
    targets, buttons = [], []
    for ys, xs in components(white_mask(rgb), 15):
        pts = P[ys, xs]; ok = np.isfinite(pts[:, 0]); pts = pts[ok]
        if len(pts) < 10: continue
        p = np.median(pts, axis=0); zmax = np.percentile(pts[:, 2], 95)
        if p[2] < -0.02 or p[2] > 0.06: continue
        if p[0] < -0.45: continue  # robot base region
        ext = np.ptp(pts[:, :2], axis=0)
        if max(ext) > 0.15: continue
        if zmax < 0.018:
            targets.append((len(xs), p, ext))
        else:
            top = pts[pts[:, 2] > zmax - 0.01]
            buttons.append((len(xs), np.median(top, axis=0), zmax, ext))
    return targets, buttons

def detect_targets(o, cam):
    """flat discs (white or red rings) on table; returns list of (npx, xyz, lit_frac)"""
    rgb = o[(cam, "rgb")]; P = world_map(o, cam)
    z = P[..., 2]
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 80) & (g < 0.45 * r) & (b < 0.45 * r)
    flat = np.isfinite(z) & (z > -0.01) & (z < 0.016)
    m = (white_mask(rgb) | red) & flat
    out = []
    for ys, xs in components(m, 15):
        pts = P[ys, xs]
        p = np.median(pts, axis=0)
        ext = np.ptp(pts[:, :2], axis=0)
        if max(ext) > 0.12 or max(ext) < 0.05: continue
        if p[0] < -0.45: continue
        lit = float(red[ys, xs].mean())
        out.append((len(xs), p, lit))
    return out

def visit(r, est, other, label, button=None, max_steps=110):
    """servo the held cube above the (possibly moving) target until it lights up."""
    est = np.array(est, float); other = np.array(other, float)
    used = 0; lit_count = 0; close_count = 0
    while used < max_steps and not r.ended:
        o = decode_obs(r.last_reply)
        cands = []
        for cam in ("front", "wrist"):
            try:
                for c, p, lit in detect_targets(o, cam):
                    if button is not None and np.linalg.norm(p[:2] - button[:2]) < 0.065:
                        continue
                    if np.linalg.norm(p[:2] - est[:2]) < 0.15 and np.linalg.norm(p[:2] - other[:2]) > 0.05:
                        cands.append((cam, c, p, lit))
            except Exception as e:
                log("detect err", e)
            if cands: break
        if cands:
            cam, c, p, lit = min(cands, key=lambda t: np.linalg.norm(t[2][:2] - est[:2]))
            if np.linalg.norm(p[:2] - est[:2]) > 0.01:
                log(label, "target moved/updated", np.round(est[:2], 3).tolist(), "->", np.round(p[:2], 3).tolist(), cam)
            est = p
            if lit > 0.15:
                lit_count += 1
            else:
                lit_count = 0
        d = np.linalg.norm(r.pos[:2] - est[:2])
        close_count = close_count + 1 if d < 0.012 else 0
        if lit_count >= 1 and d < 0.03:
            log(label, "LIT at step", r.meta["step_id"], "pos", np.round(r.pos, 3).tolist(), "est", np.round(est[:2], 3).tolist())
            return est, True
        if close_count >= 6 and not cands:
            log(label, "close (no detection) at step", r.meta["step_id"])
            return est, False
        if close_count >= 12:
            log(label, "close but not lit, step", r.meta["step_id"], "est", np.round(est[:2], 3).tolist())
            return est, False
        n = r.move_to([est[0], est[1], SWING_Z], tol=0.004, max_steps=3, chunk=3)
        if n == 0:
            r.hold(1)
            n = 1
        used += n
    log(label, "visit timeout", r.meta["step_id"])
    return est, False

def main(ctx):
    try:
        _main(ctx)
    except Exception:
        log('EXCEPTION', traceback.format_exc())
    finally:
        try:
            os.makedirs('/workspace/output', exist_ok=True)
            open('/workspace/output/controller_log.txt', 'w').write('\n'.join(LOG))
        except Exception:
            pass

def _main(ctx):
    t0 = time.time()
    r = Robot(ctx)
    ins = r.meta.get("instruction", "")
    color, n, first = parse_instruction(ins)
    log("instruction:", ins); log("parsed:", color, n, first)
    o = decode_obs(r.last_reply)
    tg, bt = find_scene(o, "wrist")
    log("wrist targets", [(c, np.round(p, 3).tolist()) for c, p, e in tg], "buttons", [(c, np.round(p, 3).tolist(), round(z, 3)) for c, p, z, e in bt])
    if len(tg) < 2 or not bt:
        tg2, bt2 = find_scene(o, "front")
        log("front targets", [(c, np.round(p, 3).tolist()) for c, p, e in tg2], "buttons", [(c, np.round(p, 3).tolist()) for c, p, z, e in bt2])
        if len(tg) < 2: tg = tg2
        if not bt: bt = bt2
    tg = sorted(tg, key=lambda t: -t[0])[:2]
    if len(tg) < 2:
        log("FAILED: could not find two targets"); return
    tgt = sorted([t[1] for t in tg], key=lambda p: p[1])
    right, left = tgt[0], tgt[1]
    button = sorted(bt, key=lambda b: -b[0])[0] if bt else None
    log("right", np.round(right, 3), "left", np.round(left, 3), "button", None if button is None else np.round(button[1], 3))

    # grasp
    for attempt in range(3):
        if grasp(r, color):
            break
        r.grip = -1.0; r.move_to(list(r.pos[:2]) + [0.15]); r.hold(2)
        r.move_to([-0.06, 0.0, 0.40]); r.hold(2)
    ests = {"right": right, "left": left}
    seq = ["right", "left"] if first == "right" else ["left", "right"]
    last = None
    for i in range(n):
        for k, side in enumerate(seq):
            if r.ended: return
            oth = "left" if side == "right" else "right"
            # travel most of the way quickly, then servo
            t = ests[side]
            ests[side], ok = visit(r, t, ests[oth], "swing %d %s" % (i + 1, side), button=None if button is None else button[1])
            t = ests[side]
            last = t
            log("swing", i + 1, "target", "first" if k == 0 else "second", "pos", np.round(r.pos, 3), "fingers", round(r.fingers, 4), "step", r.meta["step_id"])
            if r.fingers > 0.075 or r.fingers < 0.01:
                log("WARNING: cube lost?")
                # try regrasp then continue
                r.move_to([r.pos[0], r.pos[1], 0.15]); r.hold(2)
                grasp(r, color)
    # put down at last target
    if r.ended: return
    r.move_to([last[0], last[1], 0.03], speed=0.5, max_steps=40)
    r.hold(3)
    r.hold(6, grip=-1.0)
    r.move_to([last[0], last[1], 0.12], max_steps=30)
    log("put down; step", r.meta["step_id"])
    # press button
    if button is None:
        log("no button"); return
    b = np.array(button[1]); btop = button[2]
    r.grip = 1.0
    r.hold(4)
    for attempt in range(3):
        r.move_to([b[0], b[1], btop + 0.08], tol=0.004)
        r.hold(1)
        o = decode_obs(r.last_reply)
        try:
            tg3, bt3 = find_scene(o, "wrist")
        except Exception:
            bt3 = []
        near = [bb for bb in bt3 if np.linalg.norm(bb[1][:2] - b[:2]) < 0.05 and abs(bb[2] - btop) < 0.012]
        if near:
            bb = sorted(near, key=lambda x: -x[0])[0]; b = np.array(bb[1]); btop = bb[2]
            log("button refined", np.round(b, 3), round(btop, 3))
            r.move_to([b[0], b[1], btop + 0.06], tol=0.004)
        for depth in (0.012, 0.022, 0.032):
            r.move_to([b[0], b[1], btop - depth], speed=0.4, max_steps=30)
            r.hold(4)
            log("press depth", depth, "pos", np.round(r.pos, 3), "step", r.meta["step_id"], "ended", r.ended)
            if r.ended: return
            r.move_to([b[0], b[1], btop + 0.05], max_steps=20)
            r.hold(4)
            if r.ended: return
        # re-detect from high overview
        r.move_to([-0.06, 0.0, 0.45], max_steps=40); r.hold(2)
        o = decode_obs(r.last_reply)
        cands = []
        for cam in ("wrist", "front"):
            try:
                cands += [bb for bb in find_scene(o, cam)[1] if 0.02 < bb[2] < 0.06]
            except Exception:
                pass
        if cands:
            bb = min(cands, key=lambda x: np.linalg.norm(x[1][:2] - b[:2]))
            b = np.array(bb[1]); btop = bb[2]
            log("button re-detected", np.round(b, 3), round(btop, 3))
        if r.remaining < 60: break
    log("done; elapsed", round(time.time() - t0, 1))
