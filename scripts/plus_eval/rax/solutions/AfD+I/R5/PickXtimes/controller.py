"""PickXtimes controller: pick the colored cube, place it on the target N times, press the button."""
import json, math, os, re, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from perception import *
from motion import Robot, log, LOG, rotate_yaw
from skills import grasp_cube, lift, place_on_target, CUBE_HALF
from servo import Track, servo_grasp, servo_place, _det_cube, _det_target, update_target, clamp_xy, safe

WORDS = {"once": 1, "twice": 2, "thrice": 3, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
COLORS = ["green", "red", "blue", "yellow"]
CARRY_Z = 0.15


def parse_instruction(text):
    t = text.lower()
    n = None
    m = re.search(r"(\d+)\s*(?:times|x\b)", t)
    if m:
        n = int(m.group(1))
    else:
        m = re.search(r"\b(" + "|".join(WORDS) + r")\b\s*(?:times)?", t)
        # prefer the word right before 'times'
        m2 = re.search(r"\b(" + "|".join(WORDS) + r")\s+times", t)
        if m2:
            n = WORDS[m2.group(1)]
        elif re.search(r"\b(once|twice|thrice)\b", t):
            n = WORDS[re.search(r"\b(once|twice|thrice)\b", t).group(1)]
    color = "green"
    m = re.search(r"pick up the (\w+) cube", t)
    if m and m.group(1) in COLORS:
        color = m.group(1)
    return (n if n else 1), color


def scan_scene(R, color):
    """Locate target and button from both cameras at the current (high) pose."""
    tg, bt = [], []
    for cam in ("wrist", "front"):
        res, cloud = find_objects(R.meta, R.imgs, cam, color)
        if res["target"] is not None and res["target"]["n"] >= 15:
            tg.append((res["target"]["xy"], res["target"]["z"], cam))
        for b in res["button_cands"]:
            log(f"scan {cam} button cand n={b['n']} xy={b['xy'].round(3)} top={b['top']:.3f} ext={b['ext'].round(3)}")
        bc = [b for b in res["button_cands"] if b["n"] >= 15 and 0.02 < b["top"] < 0.07
              and b["xy"][0] > -0.45 and max(b["ext"]) < 0.12 and abs(b["xy"][1]) < 0.45]
        if bc:
            b = bc[0]
            bt.append((b["xy"], b["top"], cam))
    log("scan targets", [(np.round(a, 3), round(z, 3), c) for a, z, c in tg])
    log("scan buttons", [(np.round(a, 3), round(z, 3), c) for a, z, c in bt])
    return tg, bt


def refine_target(R, tgt, color):
    """Look at the target from above with an empty gripper; returns updated xy, z."""
    res, _ = find_objects(R.meta, R.imgs, "wrist", color)
    t = res["target"]
    if t is not None and t["n"] >= 30 and np.linalg.norm(t["xy"] - tgt[0]) < 0.05:
        return (t["xy"], t["z"])
    return tgt


def button_cap(R, guess_xy):
    """Find the button top surface near guess from the wrist camera."""
    res, cloud = find_objects(R.meta, R.imgs, "wrist")
    best = None
    for b in res["button_cands"]:
        if np.linalg.norm(b["xy"] - guess_xy) < 0.06 and b["n"] >= 15:
            best = b; break
    if best is None:
        return None
    # cap: highest points around the blob
    z = cloud[..., 2]
    near = np.isfinite(z) & (np.linalg.norm(cloud[..., :2] - best["xy"], axis=-1) < 0.05) & (z > 0.01) & (z < 0.08)
    pts = cloud[near]
    top = np.percentile(pts[:, 2], 98)
    cap = pts[pts[:, 2] > top - 0.004]
    return cap[:, :2].mean(0), float(top)


def press_button(R, bxy, btop):
    R.gripper(1, 8)
    R.move_to([bxy[0], bxy[1], CARRY_Z], tol=0.006)
    c = button_cap(R, bxy)
    if c is not None:
        log(f"button refined cap xy={c[0].round(3)} top={c[1]:.3f} (was {np.round(bxy,3)} {btop:.3f})")
        bxy, btop = c
    R.move_to([bxy[0], bxy[1], btop + 0.04], tol=0.004)
    # press down progressively
    for k in range(6):
        if R.ended or R.complete:
            break
        goal = btop - 0.005 - 0.004 * k
        R.move_to([bxy[0], bxy[1], goal], speed=0.006, max_iter=3)
        R.hold(3)
        log(f"press k={k} goal={goal:.3f} pos={R.pos.round(4)} complete={R.complete} ended={R.ended}")
    if not R.ended:
        R.move_to([bxy[0], bxy[1], btop + 0.06], speed=0.01, max_iter=2)
        R.hold(5)


def main(ctx):
    t0 = time.time()
    R = Robot(ctx)
    instr = R.meta.get("instruction", "")
    N, color = parse_instruction(instr)
    log(f"instruction: {instr!r} -> N={N} color={color} step={R.meta.get('step_id')} remaining={R.remaining}")
    start_done = int(os.environ.get("PICKX_DONE", "0"))  # development only
    tg, bt = scan_scene(R, color)
    if not tg:
        log("no target found!")
        tgt = (np.array([0.0, 0.0]), 0.009)
    else:
        tgt = (np.mean([a for a, _, _ in tg], 0), float(np.mean([z for _, z, _ in tg])))
    if bt:
        btn = (np.mean([a for a, _, _ in bt], 0), float(max(z for _, z, _ in bt)))
    else:
        btn = None
    log(f"target {np.round(tgt[0],3)} z={tgt[1]:.3f} button {btn}")
    # refine the target from directly above with an empty gripper
    try:
        R.gripper(-1, 3)
        R.move_to([tgt[0][0], tgt[0][1], 0.25], tol=0.008)
        res, _ = find_objects(R.meta, R.imgs, "wrist", color)
        t = res["target"]
        if t is not None and t["n"] >= 30 and np.linalg.norm(t["xy"] - tgt[0]) < 0.06:
            log(f"target refined overhead {np.round(t['xy'],3)} z={t['z']:.3f} n={t['n']}")
            tgt = (t["xy"], t["z"])
        if btn is None:
            tg2, bt2 = scan_scene(R, color)
            if bt2:
                btn = (bt2[0][0], bt2[0][1])
    except Exception as e:
        log("target refine failed", repr(e))
    done = start_done
    fails = 0
    trkT = Track("target")
    trkT.add(R.meta.get("step_id", 0), tgt[0])
    tz = tgt[1]
    # observe target motion for a few steps from the home view
    for _ in range(8):
        R.hold(2, -1.0)
        update_target(R, color, trkT, holding=False)
    p, v = trkT.predict(R.meta.get("step_id", 0))
    log(f"target track pred={np.round(p,3)} v={np.round(v,4)}")
    while done < N and not R.ended and fails < 8:
        log(f"--- cycle {done+1}/{N} step={R.meta.get('step_id')} remaining={R.remaining} t={time.time()-t0:.0f}s")
        try:
            trkC = Track("cube")
            d = _det_cube(R, color)
            if d is None:
                p = R.pos
                R.move_to([p[0], p[1], 0.2], tol=0.01)
                d = _det_cube(R, color)
            if d is None:
                fails += 1; log("cube not visible"); R.move_to([-0.06, 0.0, 0.45], tol=0.02); continue
            trkC.add(R.meta.get("step_id", 0), d[0])
            ok = servo_grasp(R, color, trkC, trkT=trkT)
            if not ok:
                fails += 1
                log("grasp failed; retry")
                R.gripper(-1, 6)
                p = R.pos
                R.move_to([p[0], p[1], 0.14], tol=0.01)
                continue
            p = R.pos
            R.move_to([p[0], p[1], 0.08], tol=0.01, max_iter=3)
            if R.state[6] + R.state[7] < 0.02:
                fails += 1; log("lost cube during lift"); R.gripper(-1, 6); continue
            placed = servo_place(R, color, trkT, tz)
            if not placed:
                fails += 1
                log("place failed; dropping cube and retrying this cycle")
                R.gripper(-1, 6)
                p = R.pos
                R.move_to([p[0], p[1], 0.12], tol=0.01, max_iter=3)
                continue
            p = R.pos
            R.move_to([p[0], p[1], 0.12], tol=0.01, max_iter=3)
            dc = _det_cube(R, color); dt = _det_target(R, color, holding=False)
            if dt is not None:
                trkT.add(R.meta.get("step_id", 0), dt[0])
            if dc is not None:
                tp, _ = trkT.predict(R.meta.get("step_id", 0))
                dist = np.linalg.norm(dc[0] - tp)
                log(f"after place: cube {np.round(dc[0],3)} target {np.round(tp,3)} dist {dist:.3f} (target seen={dt is not None})")
                _, vt = trkT.predict(R.meta.get("step_id", 0))
                if dist > 0.045 and dt is not None and np.linalg.norm(vt) < 0.001:
                    fails += 1
                    log("cube not on target; repeat without counting")
                    continue
            done += 1
        except Exception as e:
            log("cycle exception", repr(e), traceback.format_exc())
            fails += 1
            if R.ended:
                break
            try:
                R.observe()
            except Exception:
                pass
    log(f"cycles done={done} step={R.meta.get('step_id')} ended={R.ended}")
    if not R.ended and btn is not None:
        try:
            p = R.pos
            R.move_to([p[0], p[1], CARRY_Z], tol=0.01)
            press_button(R, btn[0], btn[1])
        except Exception as e:
            log("press exception", repr(e), traceback.format_exc())
    log(f"END step={R.meta.get('step_id')} complete={R.complete} ended={R.ended} t={time.time()-t0:.0f}s")
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        Path("/workspace/output/controller_log.txt").write_text("\n".join(LOG))
    except Exception:
        pass
