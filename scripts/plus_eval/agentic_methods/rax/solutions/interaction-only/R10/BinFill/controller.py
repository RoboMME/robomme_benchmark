import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lib import *

TRAVEL_Z = 0.15
HOVER_Z = 0.16
PARK = [-0.06, 0.0, 0.40]
ALLOW_PRESS = True

def save_img(R, name):
    try:
        os.makedirs("/workspace/output", exist_ok=True)
        Image.fromarray(R.imgs[("front", "rgb")][..., :3]).save(f"/workspace/output/{name}_front.png")
        Image.fromarray(R.imgs[("wrist", "rgb")][..., :3]).save(f"/workspace/output/{name}_wrist.png")
    except Exception as e:
        log("save fail", e)

def front_cubes(R):
    return [d for d in detect_cubes(R.meta, R.imgs, "front") if d["n"] >= 15]

def scan(R, gap=4):
    """Two front observations `gap` steps apart -> cubes with velocities (m/step)."""
    a = front_cubes(R); s_a = R.meta["step_id"]
    R.step([[0, 0, 0] + list(R.rot_action(R.yaw)) + [R.grip]] * gap)
    b = front_cubes(R); s_b = R.meta["step_id"]
    dt = max(s_b - s_a, 1)
    vs = []
    for d in b:
        wb = np.array(d["world"])
        best = None
        for o in a:
            if o["color"] != d["color"]: continue
            dist = np.linalg.norm(np.array(o["world"])[:2] - wb[:2])
            if dist < 0.004 * dt + 0.01 and (best is None or dist < best[0]): best = (dist, o)
        if best is not None:
            d["v"] = (wb - np.array(best[1]["world"])) / dt
            d["v"][2] = 0.0
            vs.append(d["v"])
        else:
            d["v"] = None
    vmed = np.median(np.array(vs), 0) if vs else np.zeros(3)
    for d in b:
        if d["v"] is None:
            d["v"] = vmed.copy() if d["ztop"] > 0.048 else np.zeros(3)
        if np.linalg.norm(d["v"][:2]) < 0.0004: d["v"] = np.zeros(3)
        d["s"] = s_b
    return b, vmed

def refine_bin(R, bin_c):
    R.move_to([bin_c[0] - 0.03, bin_c[1], 0.25], vmax=1.0)
    b = find_bin_full(R.meta, R.imgs, "wrist")
    if b is not None and b["n"] > 300:
        c = b["center"]
        if np.linalg.norm(c[:2] - bin_c[:2]) < 0.06:
            log("bin refined", c.round(3), "from", bin_c.round(3), "n", b["n"], "yaw", round(np.degrees(b["yaw"]), 1), "size", np.round(b["size"], 3))
            return c, b["yaw"]
    log("bin refine failed", b and b["n"])
    return bin_c, 0.0

def wrist_find(R, color, pred, radius=0.045):
    best = None
    for dd in detect_cubes(R.meta, R.imgs, "wrist"):
        if dd["color"] != color or dd["n"] < 120: continue
        ww = np.array(dd["world"])
        dist = np.linalg.norm(ww[:2] - pred[:2])
        if dist < radius and (best is None or dist < best[0]):
            best = (dist, dd)
    return best[1] if best else None

def pick(R, d, color):
    p0 = np.array(d["world"]); v = np.array(d["v"]); s0 = d["s"]; ztop = d["ztop"]
    moving = np.linalg.norm(v[:2]) > 0
    # approach hover above the (moving) cube
    for i in range(45):
        err, rerr = track_step(R, p0, v, s0, HOVER_Z, -1, vmax=1.0, vz=1.0)
        if R.ended: return False
        if err < 0.008 and i > 2: break
    # wrist refine (twice for velocity if moving)
    t = R.meta["step_id"]; pred = p0 + v * (t - s0)
    dd = wrist_find(R, color, pred)
    yaw = 0.0
    if dd is not None:
        p1 = np.array(dd["world"]); s1 = t
        yaw = cube_yaw(R.meta, R.imgs, "wrist", dd["px"], color)
        ztop = dd["ztop"]
        p0, s0 = p1, s1
        log("wrist refine", np.round(p0, 3), "yaw", round(np.degrees(yaw), 1), "ztop", round(ztop, 3), "v", np.round(v * 1000, 2))
    else:
        log("wrist refine failed; pred", np.round(pred, 3))
    R.yaw = yaw
    for i in range(25):
        err, rerr = track_step(R, p0, v, s0, HOVER_Z, -1, vmax=1.0, vz=1.0)
        if R.ended: return False
        if err < 0.005 and rerr < 0.06: break
    # second wrist refine after rotation
    t = R.meta["step_id"]; pred = p0 + v * (t - s0)
    dd = wrist_find(R, color, pred, radius=0.03)
    if dd is not None:
        p0 = np.array(dd["world"]); s0 = t; ztop = dd["ztop"]
    gz = max(ztop - 0.022, 0.012)
    for i in range(40):
        err, rerr = track_step(R, p0, v, s0, gz, -1, vmax=1.0, vz=0.5)
        if R.ended: return False
        if err < 0.005: break
    log("at grasp", R.state[:3].round(3), "err", round(err, 4))
    for i in range(10):
        track_step(R, p0, v, s0, gz, 1, vmax=1.0, vz=0.3)
        if R.ended: return False
    width = R.state[6] + R.state[7]
    log("closed width", round(width, 4))
    R.move_to(list(R.state[:2]) + [TRAVEL_Z], grip=1, vmax=0.6, tol=0.01)
    width = R.state[6] + R.state[7]
    ok = width > 0.025
    held = {}
    try:
        mk = color_masks(R.imgs[("wrist", "rgb")])
        held = {k: int(m[150:, :].sum()) for k, m in mk.items()}
        if ok and held.get(color, 0) < 600 and max(held.values()) > 1500:
            log("WRONG color held", held, "-> discard")
            R.move_to(list(R.state[:2]) + [0.08], grip=1, vmax=0.6, tol=0.01, max_iter=15)
            R.gripper(-1, 6)
            R.move_to(list(R.state[:2]) + [TRAVEL_Z], grip=-1, vmax=0.6, tol=0.01, max_iter=15)
            return False
    except Exception as e:
        log("held check error", e)
    log("after lift width", round(width, 4), "ok", ok, "held px", held)
    return ok

def place(R, bin_c, bin_yaw=0.0):
    k = np.round((R.yaw - bin_yaw) / (np.pi / 2))
    R.move_to([bin_c[0], bin_c[1], TRAVEL_Z], grip=1, vmax=0.8, tol=0.006, yaw=bin_yaw + k * np.pi / 2)
    R.move_to([bin_c[0], bin_c[1], bin_c[2] + 0.075], vmax=0.4, tol=0.004)
    R.gripper(-1, 8)
    R.move_to([bin_c[0], bin_c[1], TRAVEL_Z], vmax=0.8, tol=0.01)

def press_button(R, btn):
    c = btn["center"]; top = btn["ztop"]
    R.move_to([c[0], c[1], TRAVEL_Z], grip=1, vmax=1.0, tol=0.006, yaw=0.0)
    R.move_to([c[0], c[1], top + 0.03], vmax=0.5, tol=0.004)
    for i in range(30):
        if R.ended: break
        e = (top - 0.02) - R.state[2]
        R.step([[0, 0, float(np.clip(e / 0.05, -0.3, 0.3)), 0, 0, 0, 1]])
    log("press bottom z", R.state[2].round(3), "ended", R.ended, "complete", R.complete)
    if not R.ended:
        R.move_to([c[0], c[1], top + 0.08], vmax=0.5, tol=0.01, max_iter=15)

def park(R):
    R.move_to(PARK, vmax=1.0, tol=0.02, max_iter=20)

def choose(R, cubes, todo, bin_c):
    cands = []
    for d in cubes:
        if d["color"] not in todo: continue
        w = np.array(d["world"]); v = d["v"]
        if d["ztop"] > 0.075: continue
        if w[0] < -0.40 or abs(w[1]) > 0.40: continue
        if np.linalg.norm(w[:2] - bin_c[:2]) < 0.05: continue
        fut = w + v * 70
        if np.linalg.norm(v[:2]) > 0:
            # must stay within reachable band for ~70 steps (conveyor carries it away)
            if abs(fut[1]) > 0.30 or fut[0] < -0.40 or fut[0] > 0.2: continue
        others = [np.array(o["world"]) for o in cubes if o is not d]
        clear = min([np.linalg.norm(w[:2] - o[:2]) for o in others] + [1.0])
        cost = 0.5 * np.linalg.norm(w[:2] - R.state[:2]) - 2.0 * min(clear, 0.07)
        cands.append((cost, d, clear))
    cands.sort(key=lambda c: c[0])
    return cands

def run(ctx, need=None, allow_press=ALLOW_PRESS):
    R = Robot(ctx)
    R.observe()
    instr = R.meta.get("instruction", "")
    log("instruction:", instr)
    if need is None:
        need = parse_instruction(instr)
    log("need", need)
    bin0 = find_bin(R.meta, R.imgs, "front")
    if bin0 is None:
        bin0 = find_bin(R.meta, R.imgs, "front", dark=40)
        log("bin fallback (dark<40):", bin0 is not None)
    bin_c = (bin0["lo"] + bin0["hi"]) / 2 if bin0 else np.array([0.03, -0.15, 0.026])
    btn = find_button(R.meta, R.imgs, "front", bin_xy=bin_c)
    if btn is None:
        btn = find_button(R.meta, R.imgs, "front", bin_xy=bin_c, satmax=12, xmin=-0.6)
        log("button fallback:", btn is not None)
    log("bin", bin_c.round(3), "button", btn and np.round(btn["center"], 3))
    bin_c, bin_yaw = refine_bin(R, bin_c)
    placed = {k: 0 for k in need}
    failures = 0
    t0 = time.time()
    while not R.ended and failures < 12:
        todo = [c for c in need if placed[c] < need[c]]
        if not todo: break
        if R.meta["episode_limits"]["remaining_episode_steps"] < 150: log("low steps"); break
        if time.time() - t0 > 950: log("time"); break
        park(R)
        cubes, vmed = scan(R)
        cands = choose(R, cubes, todo, bin_c)
        if not cands:
            log("no candidates for", todo, "step", R.meta["step_id"], "; waiting")
            R.step([[0, 0, 0, 0, 0, 0, -1]] * 6)
            continue
        cost, d, clear = cands[0]
        log(f"pick {d['color']} at {np.round(d['world'],3)} v {np.round(d['v']*1000,2)} clear {clear:.3f} placed {placed} step {R.meta['step_id']}")
        ok = pick(R, d, d["color"])
        if R.ended: break
        if not ok:
            failures += 1
            R.gripper(-1, 4)
            continue
        place(R, bin_c, bin_yaw)
        placed[d["color"]] += 1
        log("placed", placed, "step", R.meta["step_id"])
    if not R.ended and allow_press:
        park(R)
        btn2 = find_button(R.meta, R.imgs, "front", bin_xy=bin_c)
        if btn2 is None and btn is None:
            btn2 = find_button(R.meta, R.imgs, "front", bin_xy=bin_c, satmax=12, xmin=-0.6)
        if btn2 is not None and (btn is None or np.linalg.norm(btn2["center"][:2] - btn["center"][:2]) < 0.05):
            btn = btn2
        if btn is not None:
            press_button(R, btn)
    log("final placed", placed, "ended", R.ended, "complete", R.complete, "step", R.meta.get("step_id"))
    save_img(R, "final")
    return R

def main(ctx):
    try:
        run(ctx)
    except Exception:
        log("EXCEPTION", traceback.format_exc())
