"""ButtonUnmaskSwap controller.

1. step 0: memorise coloured cubes (front RGB-D) and button positions.
2. step until the 8 containers drop; initialise a tracker (container <-> cube by proximity).
3. press each button quickly while tracking every saved frame.
4. retract, keep tracking until the shuffle stops.
5. lift the containers hiding the instructed colours in order (place each back down).
"""
import base64, io, json, os, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from PIL import Image

from percep import depth_to_world, blobs_from_points, detect_cubes, raster_blobs, static_mask_from
ZLO = 0.062
MINB = 15
from tracker import Tracker
from robot import Robot, rot_err, packet, rotate_to, R_DOWN
from percep import square_yaw
from logic import parse_order, top_center

ROOT = Path("/workspace")
OUT = ROOT / "output"
HOME = np.array([-0.06, 0.0, 0.50])


class Log:
    def __init__(self):
        self.t0 = time.time()
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            self.f = open(OUT / "controller_log.txt", "a")
        except Exception:
            self.f = None

    def __call__(self, *a):
        s = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        if self.f:
            try:
                self.f.write(s + "\n"); self.f.flush()
            except Exception:
                pass


log = Log()


def geo_from_frame(frame, cam='front'):
    fp = {(x['camera'], x['kind']): x['path'] for x in frame['files']}
    d = np.array(Image.open(ROOT / fp[(cam, 'depth')])).astype(np.float64) * 0.001
    rgb = np.array(Image.open(ROOT / fp[(cam, 'rgb')]).convert("RGB"))
    P = depth_to_world(d, frame['camera_intrinsics'][cam]['matrix'], frame['camera_poses'][cam])
    return rgb, P, d > 0


def merge_objs(primary, secondary, r=0.03):
    out = list(primary)
    for o in secondary:
        if all(np.linalg.norm(np.array(o['xy']) - np.array(p['xy'])) > r for p in out):
            out.append(o)
    return out


def geo_from_reply(blocks, pkt):
    imgs = {}
    for a in pkt.get("attachments", []):
        b = blocks[a["content_index"]]
        imgs[(a["camera"], a["kind"])] = Image.open(io.BytesIO(base64.b64decode(b["data"])))
    mpu = pkt.get("depth", {}).get("front", {}).get("meters_per_unit", 0.001)
    d = np.array(imgs[("front", "depth")]).astype(np.float64) * mpu
    rgb = np.array(imgs[("front", "rgb")].convert("RGB"))
    P = depth_to_world(d, pkt['camera_intrinsics']['front']['matrix'], pkt['camera_poses']['front'])
    return rgb, P, d > 0


class Ctl:
    def __init__(self, ctx):
        self.ctx = ctx
        self.r = Robot(ctx, log=log)
        self.r.step = self._step_wrap(self.r.step)
        self.tracker = None
        self.seen_steps = set()
        self.last_geo = None
        self.blob_hist = []  # (step, blobs)
        self.smask = None
        self.last_wrist = None
        self.frozen = False

    # ---- frames -------------------------------------------------------
    def _frames_of(self, pkt):
        fs = pkt.get("frame_sequence") or {}
        mp = fs.get("manifest_path")
        if not mp:
            return []
        try:
            return json.loads((ROOT / mp).read_text()).get("frames", [])
        except Exception as e:
            log("manifest read failed", e)
            return []

    def _step_wrap(self, orig):
        def step(rows):
            p = orig(rows)
            tr = p.get("transition") or {}
            if tr.get("task_complete") or p.get("task_complete"):
                self.r.task_complete = True
            if tr.get("episode_ended"):
                self.r.ended = True
            try:
                self.process(p)
            except Exception as e:
                log("process error", repr(e))
            return p
        return step

    def process(self, pkt, blocks=None):
        frames = self._frames_of(pkt)
        got = False
        for f in frames:
            s = f.get("step_id")
            if s in self.seen_steps or f.get("phase") not in (None, "execution"):
                continue
            try:
                g = geo_from_frame(f)
                gw = geo_from_frame(f, 'wrist')
            except Exception as e:
                log("frame load failed", s, e)
                continue
            self.seen_steps.add(s)
            self.last_wrist = (s, gw, f.get('state'))
            self.on_geo(s, g, gw, f.get('state'))
            got = True
        if not got and pkt.get("step_id") not in self.seen_steps and blocks is not None:
            try:
                g = geo_from_reply(blocks, pkt)
                self.seen_steps.add(pkt["step_id"])
                self.on_geo(pkt["step_id"], g)
            except Exception as e:
                log("inline decode failed", e)

    def on_geo(self, step, g, gw=None, state=None):
        rgb, P, V = g
        self.last_geo = (step, g)
        sets = [(P, V)] + ([(gw[1], gw[2])] if gw is not None else [])
        exc = None
        if state is not None and state[2] < 0.2:
            exc = [((state[0], state[1]), 0.045)]
        bl, _ = raster_blobs(sets, zlo=ZLO, static_mask=self.smask, exclude=exc)
        bl = [b for b in bl if b['n'] >= MINB]
        self.blob_hist.append((step, bl))
        if self.tracker is not None and not self.frozen:
            k = self.tracker.update(step, bl)
            if k < len(self.tracker.p):
                log(f"step {step}: matched {k}/{len(self.tracker.p)} blobs={len(bl)}")

    # ---- phases -------------------------------------------------------
    def run(self):
        r = self.r
        reply = self.ctx.call("robomme_observe")
        pkt = packet(reply)
        r.pkt = pkt
        instr = pkt.get("instruction", "")
        order = parse_order(instr)
        log("instruction:", instr)
        log("order:", order, "step", pkt["step_id"])
        # conditioning frames (if any): use the execution frame for cubes
        frames = [f for f in self._frames_of(pkt) if f.get("phase") == "execution"]
        objs_w = []
        if frames:
            g0 = geo_from_frame(frames[-1])
            try:
                gw0 = geo_from_frame(frames[-1], 'wrist')
                objs_w = detect_cubes(*gw0)
                self.smask = static_mask_from([(g0[1], g0[2]), (gw0[1], gw0[2])], zlo=ZLO)
                log("static mask cells", int(self.smask.sum()))
            except Exception as e:
                log("wrist cube detection failed", e)
        else:
            g0 = geo_from_reply(reply["content"] if isinstance(reply, dict) else reply, pkt)
        self.seen_steps.add(pkt["step_id"])
        rgb, P, V = g0
        objs_f = detect_cubes(rgb, P, V)
        objs_w = [o for o in objs_w if o['ztop'] < 0.045 and o['n'] >= 20]
        objs_f = [o for o in objs_f if o['ztop'] < 0.045]
        objs = merge_objs(objs_w, objs_f)
        log("step0 objects:", [(o['color'], np.round(o['xy'], 3).tolist(), o['n']) for o in objs])

        # wait for containers to appear
        n_cont = None
        prev_n = -1
        for i in range(40):
            r.hold(1, grip=-1.0)
            st, bl = self.blob_hist[-1] if self.blob_hist else (None, [])
            full = [b for b in bl if b['n'] >= 50]
            if len(full) >= 3 and len(full) == prev_n and st >= 7:
                n_cont = len(full)
                break
            prev_n = len(full)
        if not n_cont:
            log("containers not detected; using last blobs")
            st, bl = self.blob_hist[-1]
            full = bl
        self.tracker = Tracker([b['xy'] for b in full])
        self.tracker.t = st
        init = self.tracker.p.copy()
        log(f"tracker init step {st} with {len(full)} containers")

        # cube -> container, buttons = objects not under a container
        cube_of = {}
        buttons = []
        for o in objs:
            dd = np.linalg.norm(init - np.array(o['xy']), axis=1)
            j = int(dd.argmin())
            if dd[j] < 0.045:
                cube_of.setdefault(o['color'], []).append(j)
            else:
                buttons.append(o)
        log("cube_of:", cube_of)
        btn_xy = [np.array(b['xy']) for b in buttons if b['xy'][0] < 0.0]
        if len(btn_xy) > 2:
            btn_xy = sorted(btn_xy, key=lambda p: p[0])[:2]
        btn_xy = sorted(btn_xy, key=lambda p: p[1])
        log("buttons:", [b.round(3).tolist() for b in btn_xy])

        # press buttons
        for b in btn_xy:
            if r.ended: break
            r.move_to([b[0], b[1], 0.07], grip=1.0, max_steps=30, chunk=5, tol=0.01)
            r.move_to([b[0], b[1], 0.012], grip=1.0, max_steps=12, chunk=4, vmax=0.8)
            log(f"pressed {b.round(3)} ee={r.state[:3].round(3)} step {r.step_id}")
            r.move_to([b[0], b[1], 0.07], grip=1.0, max_steps=6, chunk=3, tol=0.01)
        r.move_to(HOME, grip=1.0, max_steps=30, chunk=5)
        log("home at step", r.step_id)

        # wait until the shuffle stops
        still = 0
        prev = self.tracker.p.copy()
        lim = int(self.r.pkt.get('episode_limits', {}).get('episode_step_limit', 1312))
        cap = max(300, min(lim - 450, 1400))
        log('settle cap', cap)
        while not r.ended and r.step_id < cap:
            r.hold(5)
            mv = float(np.median(np.linalg.norm(self.tracker.p - prev, axis=1)))
            mx = float(np.max(np.linalg.norm(self.tracker.p - prev, axis=1)))
            prev = self.tracker.p.copy()
            still = still + 1 if (mv < 0.002 and mx < 0.006) else 0
            if still >= 2 and r.step_id >= 185:
                break
        self.frozen = True
        log("settled at step", r.step_id, "positions", self.tracker.p.round(3).tolist(), "reacq", self.tracker.reacq)

        # pick in order
        for k, color in enumerate(order):
            if r.ended: break
            idx = cube_of.get(color)
            if not idx:
                log("no container for", color, "- skipping"); continue
            j = idx[0]
            xy = np.array(self.tracker.p[j])
            fing = 0.0
            for attempt in range(2):
                xy, yaw, ok = self.inspect(xy)
                log(f"target {color}: track {j} -> xy {np.round(xy, 3)} yaw {np.degrees(yaw):.0f} ok {ok}")
                rotate_to(r, yaw)
                fing = pick(r, xy)
                if r.ended or fing > 0.01:
                    break
                log("grasp failed; reopen and re-inspect")
                r.hold(4, grip=-1.0)
                r.move_to([xy[0], xy[1], 0.20], grip=-1.0, max_steps=20, chunk=5)
            if r.ended: break
            if k < len(order) - 1:
                place(r, xy)
        log("done: ended", r.ended, "task_complete", r.task_complete, "step", r.step_id)


def _inspect(self, xy):
    """hover over xy, use wrist depth top-face points to refine centre and yaw"""
    r = self.r
    r.R_target = R_DOWN
    # wrist camera sits offset from the TCP; hover at 0.22 to see the top face
    r.move_to([xy[0], xy[1], 0.22], grip=-1.0, max_steps=50, chunk=5, tol=0.006)
    r.hold(2, grip=-1.0)
    if not self.last_wrist:
        return np.array(xy), 0.0, False
    s, gw, st = self.last_wrist
    rgb, P, V = gw
    best_xy, best_yaw, ok = np.array(xy), 0.0, False
    for rad in (0.03, 0.04):
        m = V & (P[..., 2] > 0.066) & (P[..., 2] < 0.085) & (np.linalg.norm(P[..., :2] - best_xy, axis=-1) < rad)
        if m.sum() < 30:
            continue
        pts = P[m][:, :2]
        best_xy = np.median(pts, axis=0)
    m = V & (P[..., 2] > 0.066) & (P[..., 2] < 0.085) & (np.linalg.norm(P[..., :2] - best_xy, axis=-1) < 0.035)
    if m.sum() >= 30:
        pts = P[m][:, :2]
        best_yaw = square_yaw(pts)
        ok = True
        if np.linalg.norm(best_xy - np.array(xy)) > 0.035:
            log("inspect: refined centre far from track; keeping track xy")
            best_xy = np.array(xy); ok = False
    return best_xy, best_yaw, ok

Ctl.inspect = _inspect


def pick(r, xy, z_grasp=0.058):
    x, y = float(xy[0]), float(xy[1])
    r.move_to([x, y, z_grasp + 0.03], grip=-1.0, max_steps=20, chunk=4, vmax=0.6)
    r.move_to([x, y, z_grasp], grip=-1.0, max_steps=15, chunk=3, vmax=0.3)
    if r.ended: return 0.0
    r.hold(8, grip=1.0)
    r.move_to([x, y, 0.20], grip=1.0, max_steps=30, chunk=4, vmax=0.6)
    if r.ended: return 1.0
    r.hold(2)
    fing = float(r.state[6] + r.state[7])
    log(f"pick at {x:.3f},{y:.3f}: ee={r.state[:3].round(3)} opening={fing:.4f} step={r.step_id} ended={r.ended} complete={r.task_complete}")
    return fing


def place(r, xy, z_place=0.062):
    x, y = float(xy[0]), float(xy[1])
    r.move_to([x, y, z_place + 0.04], grip=1.0, max_steps=40, chunk=4)
    r.move_to([x, y, z_place], grip=1.0, max_steps=15, chunk=3, vmax=0.3)
    r.hold(6, grip=-1.0)
    r.move_to([x, y, 0.16], grip=-1.0, max_steps=20, chunk=4)
    log(f"placed at {x:.3f},{y:.3f} step={r.step_id}")


def main(ctx):
    c = Ctl(ctx)
    try:
        c.run()
    except Exception:
        log("FATAL", traceback.format_exc())
