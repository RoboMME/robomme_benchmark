"""BinFill controller v1: put requested coloured cubes in the bin, then press the button."""
import os, re, sys, time, traceback, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion import Robot
from percep import find_cubes, find_bin, find_button
from skills import pick, drop_in_bin

VERSION = "v5"
PRESET = {}  # dev only: cubes already placed
HOME = [-0.06, 0.0, 0.45]
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
COLORS = ("red", "green", "blue")

def parse(instr):
    need = {}
    for n, c in re.findall(r"\b(\d+|one|two|three|four|five|six|a|an)\s+(red|green|blue)\s+cubes?", instr.lower()):
        need[c] = need.get(c, 0) + (int(n) if n.isdigit() else NUM[n])
    return need

class Log:
    def __init__(self):
        os.makedirs("/workspace/output", exist_ok=True)
        self.f = open("/workspace/output/controller_log.txt", "a")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        try:
            self.f.write(s+"\n"); self.f.flush()
        except Exception:
            pass

def press_button(rb, btn, log):
    x, y = btn["xyz"][0], btn["xyz"][1]; zt = btn["ztop"]
    rb.move_to([x, y, 0.12], grip=1, vmax=0.6)
    rb.gripper(1, 4)
    rb.move_to([x, y, zt+0.03], vmax=0.3)
    for dz in (0.0, -0.01, -0.02, -0.03):
        rb.move_to([x, y, zt+dz], vmax=0.15, max_iter=4)
        log(f"press: z={rb.state[2]:.3f} ended={rb.ended} complete={rb.complete}")
        if rb.ended:
            return
    rb.step([[0, 0, -0.1, 0, 0, 0, 1]]*5)
    rb.move_to([x, y, 0.12], vmax=0.4)

def mat_extent(m, im, v):
    """Max projection of low-saturation raised surface (conveyor mat) along unit v."""
    from rio import pointcloud
    P = pointcloud(m, im, "front"); rgb = im[("front", "rgb")].astype(int)
    sat = rgb.max(-1) - rgb.min(-1)
    msk = (sat < 15) & (P[..., 2] > 0.005) & (P[..., 2] < 0.03) & (P[..., 0] > -0.3)
    pts = P[msk]
    if len(pts) < 200:
        return None
    return float(np.percentile(pts[:, :2] @ v, 99))

def scan(rb, log, st):
    if st.get("fast") and np.linalg.norm(st["vel"]) > 0:
        # conveyor + bin behind the mat: observe from where we are (over the bin)
        m, im = rb.observe()
        cubes = find_cubes(m, im, "front")
        b = find_bin(m, im)
        btn = find_button(m, im, bin_xy=None if b is None else b["xyz"])
        return b, btn, cubes, rb.sid()
    rb.move_to(HOME, vmax=1.0, tol=0.02)
    rb.step([[0, 0, 0, 0, 0, 0, rb.grip]]*2)
    mA, imA, tA = rb.meta, rb.imgs, rb.sid()
    cA = find_cubes(mA, imA, "front")
    rb.step([[0, 0, 0, 0, 0, 0, rb.grip]]*5)
    m, im, tB = rb.meta, rb.imgs, rb.sid()
    cubes = find_cubes(m, im, "front")
    ds = []
    for c in cubes:
        best = None
        for a in cA:
            if a["color"] != c["color"]:
                continue
            d = c["xyz"][:2] - a["xyz"][:2]
            if np.linalg.norm(d) < 0.04 and (best is None or np.linalg.norm(d) < np.linalg.norm(best)):
                best = d
        if best is not None:
            ds.append(best)
    if len(ds) >= 1 and tB > tA:
        v = np.median(np.array(ds), 0) / (tB - tA)
        st["vel"] = np.array([v[0], v[1], 0.0]) if np.linalg.norm(v) > 3e-4 else np.zeros(3)
    b = find_bin(m, im)
    btn = find_button(m, im, bin_xy=None if b is None else b["xyz"])
    st["mat_end"] = None
    if np.linalg.norm(st["vel"]) > 0:
        vu = st["vel"][:2]/np.linalg.norm(st["vel"][:2])
        st["mat_end"] = mat_extent(m, im, vu)
    return b, btn, cubes, tB

def main(ctx):
    log = Log()
    rb = Robot(ctx, log)
    rb.observe()
    instr = rb.meta.get("instruction", "")
    need = parse(instr)
    log(f"[{VERSION}] instruction={instr!r} need={need}")
    placed = {c: int(PRESET.get(c, 0)) for c in need}
    bin_xyz = None; btn = None
    fails = {}
    st = {"vel": np.zeros(3), "mat_end": None}
    t0 = time.time()
    while not rb.ended:
        try:
            todo = [c for c in need if placed[c] < need[c]]
            if not todo:
                break
            if time.time()-t0 > 1000 or rb.remaining() < 160:
                log("budget low; go press"); break
            b, bt, cubes, tS = scan(rb, log, st)
            vel = st["vel"]; speed = float(np.linalg.norm(vel))
            if b is not None and (bin_xyz is None or b["npx"] > 100):
                bin_xyz = b["xyz"]
            if bt is not None:
                btn = bt
            st["fast"] = bin_xyz is not None and bin_xyz[0] < -0.18
            if bin_xyz is None:
                log("no bin seen"); rb.step([[0, 0, 0, 0, 0, 0, -1]]*10); continue
            cand = []
            for c in cubes:
                if c["color"] not in todo:
                    continue
                if np.hypot(*(c["xyz"][:2]-bin_xyz[:2])) < 0.07:
                    continue
                key = tuple(np.round(c["xyz"][:2], 2)) if speed == 0 else (c["color"], len(fails))
                if fails.get(key, 0) >= 2:
                    continue
                others = [o for o in cubes if o is not c and np.linalg.norm(o["xyz"][:2]-c["xyz"][:2]) < 0.06]
                if speed > 0:
                    vu = vel[:2]/speed
                    end = st["mat_end"] if st["mat_end"] is not None else float(c["xyz"][:2] @ vu) + 0.3
                    life = (end - 0.02 - float(c["xyz"][:2] @ vu)) / speed
                    travel = np.linalg.norm(c["xyz"][:2] - rb.state[:2]) / 0.03
                    if life < travel + 28:
                        continue
                    score = life
                else:
                    score = np.hypot(*(c["xyz"][:2]-bin_xyz[:2])) + 0.15*len(others)
                cand.append((score, key, c))
            log(f"step={rb.meta['step_id']} vel={np.round(vel,4)} mat_end={st['mat_end']} placed={placed} todo={todo} cands={[(k, round(s,3)) for s,k,_ in sorted(cand, key=lambda t:t[0])]}")
            if not cand:
                rb.step([[0, 0, 0, 0, 0, 0, -1]]*(4 if speed > 0 else 8)); continue
            cand.sort(key=lambda t: t[0])
            _, key, c = cand[0]
            ok = pick(rb, c["color"], c["xyz"], tS, vel, log)
            if not ok:
                fails[key] = fails.get(key, 0)+1
                rb.gripper(-1, 6)
                continue
            held = drop_in_bin(rb, bin_xyz, log)
            if held:
                placed[c["color"]] += 1
            log(f"placed={placed}")
        except Exception as e:
            log("ERR", repr(e), traceback.format_exc())
            if rb.ended:
                break
            try:
                rb.observe()
            except Exception:
                pass
    if not rb.ended:
        try:
            if btn is None:
                _, btn, _ = scan(rb, log)
            if btn is not None:
                press_button(rb, btn, log)
            if not rb.ended:
                rb.step([[0, 0, 0, 0, 0, 0, 1]]*5)
        except Exception as e:
            log("ERR press", repr(e), traceback.format_exc())
    log(f"done step={rb.meta.get('step_id')} ended={rb.ended} complete={rb.complete} placed={placed}")
