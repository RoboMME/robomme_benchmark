"""MoveCube controller (RAX, interaction-only, R=1): blind-built, observation-driven.

Pipeline / transition table
  INIT      observe step 0, status limits                       -> VIDEO (always)
  VIDEO     read conditioning manifest, classify manner          -> SURVEY (default manner 'pick' on failure)
  SURVEY    front RGB-D: table, cube, tool, target               -> PROBE ; no cube/target -> SURVEY_HIGH
  SURVEY_HIGH  lift wrist camera over table, retry survey       -> PROBE or STOP
  PROBE     measure action->motion map, rotation composition     -> EXECUTE
  EXECUTE   pick_place | push_gripper | tool_use (each retries)  -> VERIFY
  VERIFY    relocate cube; if far from target and budget left    -> EXECUTE (max 2 rounds) else STOP
  STOP      lift clear, return
Every phase is wrapped: exceptions are logged and lead to the next rung; EpisodeEnded/OutOfBudget stop.
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import mcio  # noqa: E402
import video  # noqa: E402
from motion import EpisodeEnded, OutOfBudget, Robot  # noqa: E402
from perception import Obs, Perceiver  # noqa: E402
from scene import Scene  # noqa: E402
from tasks import Executor  # noqa: E402

VERSION = "movecube-v1"
WALL_LIMIT = 1050.0          # s, controller budget (runtime hard limit 1200 s)
OUT_DIR = "/workspace/output"


class Logger:
    def __init__(self):
        self.f = None
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            self.f = open(os.path.join(OUT_DIR, "controller_log.txt"), "a")
        except Exception:
            self.f = None
        self.t0 = time.time()

    def __call__(self, *a):
        msg = f"[{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if self.f:
            try:
                self.f.write(msg + "\n")
                self.f.flush()
            except Exception:
                pass


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k not in ("pts", "cols", "pix", "top", "tool_pts")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def describe_reply(reply, log):
    try:
        m = mcio.meta(reply)
        keys = sorted(m.keys())
        log("reply keys:", keys)
        kinds = [(b.get("type"), (b.get("mimeType") or ""), len(b.get("data", "") or b.get("text", "")))
                 for b in mcio.blocks(reply)]
        log("reply blocks:", kinds[:12])
        small = {k: v for k, v in m.items() if k not in ("frame_sequence",) and len(json.dumps(v, default=str)) < 600}
        log("reply small fields:", json.dumps(small, default=str)[:3000])
    except Exception as e:
        log("describe_reply failed", repr(e))


def analyse_video(R, per, scene, obs0, log):
    manner, info = None, {}
    prior = video.instruction_prior(mcio.get_instruction(obs0.meta))
    sf, imf = {}, {}
    try:
        man = mcio.read_manifest(obs0.meta)
        if man is None:
            log("video: no manifest")
        else:
            vid, exe = video.load_video_frames(man)
            log(f"video: {len(vid)} video frames, {len(exe)} execution frames")
            sf = video.state_features(vid, scene.table_z)
            log("video state features:", json.dumps(_jsonable({k: v for k, v in sf.items() if k != "segments"}), default=str)[:1500])
            log("video segments:", json.dumps(_jsonable(sf.get("segments", [])), default=str)[:1500])
            var = per.variant.get("front")
            imf = video.image_features(vid, per, obs0.K.get("front"), obs0.pose.get("front"), var, scene.table_z, log=log)
            log("video image features:", json.dumps(_jsonable({k: v for k, v in imf.items() if k not in ("cube_track", "tool_track")}), default=str)[:1500])
            log("video cube track:", json.dumps(_jsonable(imf.get("cube_track", [])), default=str)[:1500])
            log("video tool track:", json.dumps(_jsonable(imf.get("tool_track", [])), default=str)[:800])
    except Exception:
        log("video analysis failed:", traceback.format_exc()[-1500:])
    try:
        manner, votes, why = video.decide(sf, imf, prior, log=log)
    except Exception:
        log("video decide failed:", traceback.format_exc()[-800:])
        manner = prior or "pick"
    return manner, sf, imf


def main(ctx):
    log = Logger()
    t0 = time.time()
    log(f"controller {VERSION} start")
    R = Robot(ctx, log=log, t0=t0, wall_limit=WALL_LIMIT)
    try:
        _run(ctx, R, log)
    except (EpisodeEnded, OutOfBudget) as e:
        log("stop:", type(e).__name__, str(e))
    except Exception:
        log("FATAL (caught):", traceback.format_exc()[-3000:])
    log(f"controller end: steps sent {R.steps_sent}, calls {R.calls}, wall {time.time() - t0:.1f}s, "
        f"ended={R.ended} complete={R.complete}")


def _run(ctx, R, log):
    r0 = R.observe()
    describe_reply(r0, log)
    st = R.status()
    try:
        log("status small:", json.dumps({k: v for k, v in st.items() if len(json.dumps(v, default=str)) < 400}, default=str)[:2000])
    except Exception:
        pass
    log(f"limits: max_steps={R.max_steps} remaining={R.remaining} step={R.step_id}")
    obs0 = Obs(r0)
    log("obs0:", json.dumps(obs0.summary(), default=str))
    instr = mcio.get_instruction(obs0.meta)
    log("instruction:", instr)
    if R.state is None:
        log("no state; cannot control")
        return
    per = Perceiver(ctx, log=log)
    scene = Scene(R, per, log=log)
    try:
        from scene import instruction_color
        scene.want_color = instruction_color(instr)
        if scene.want_color is not None:
            log("instruction colour", scene.want_color)
    except Exception:
        pass
    # perception at step 0 (robot at rest)
    try:
        per.calibrate(obs0, "front")
        scene.survey()
    except (EpisodeEnded, OutOfBudget):
        raise
    except Exception:
        log("survey failed:", traceback.format_exc()[-1500:])
    manner, sf, imf = analyse_video(R, per, scene, obs0, log)
    log(f"MANNER = {manner}")
    reconcile_target(scene, imf, log)
    R.setup_orientation()
    if manner == "tool":
        infer_closing_axis(R, sf, imf, log)
    try:
        R.probe(grip=-1.0)
    except (EpisodeEnded, OutOfBudget):
        raise
    except Exception:
        log("probe failed:", traceback.format_exc()[-1000:])
    if scene.cube is None or scene.target is None or (manner == "tool" and scene.tool is None):
        survey_high(R, scene, log)
    if scene.cube is None or scene.target is None:
        log("STOP: missing cube or target after surveys")
        return
    if manner == "tool" and scene.tool is None:
        log("tool manner but no tool found; falling back to push")
        manner = "push"
    ex = Executor(R, scene, log=log)
    for rnd in range(3):
        ok = False
        try:
            log(f"[phase] execute {manner} round {rnd} (step {R.step_id}, wall {R.wall():.0f}s)")
            if manner == "pick":
                ok = ex.pick_place()
            elif manner == "push":
                ok = ex.push_gripper()
            else:
                ok = ex.tool_use()
        except (EpisodeEnded, OutOfBudget):
            raise
        except Exception:
            log("execute error:", traceback.format_exc()[-2000:])
        log(f"execute round {rnd} result {ok}; complete={R.complete}")
        if ok or R.complete:
            break
        if R.steps_left() < 40:
            break
        try:
            scene.survey()
        except (EpisodeEnded, OutOfBudget):
            raise
        except Exception:
            log("resurvey failed:", traceback.format_exc()[-800:])
    # finish: stay clear, gripper state per manner
    try:
        grip = 1.0 if manner in ("push", "tool") else -1.0
        p = R.pos.copy()
        if scene.table_z is not None and p[2] < scene.table_z + 0.12:
            p[2] = scene.table_z + 0.12
            R.move_to(p, None, grip, tol=0.01, max_iter=4, label="final lift")
        R.send([[0, 0, 0, 0, 0, 0, grip]] * 3)
    except (EpisodeEnded, OutOfBudget):
        raise
    except Exception:
        log("final lift failed:", traceback.format_exc()[-500:])


def infer_closing_axis(R, sf, imf, log):
    """From the video's tool grasp: the horizontal gripper axis most perpendicular to the tool is the closing axis."""
    try:
        import geom
        segs = [s for s in sf.get("segments", []) if s.get("width", 0) > 0.004]
        tp = imf.get("tool_pts")
        if not segs or tp is None:
            return
        seg = max(segs, key=lambda s: s["b"] - s["a"])
        g = np.array(seg["xy_grasp"])
        near = tp[np.linalg.norm(tp - g, axis=1) < 0.04]
        if len(near) < 5 or np.min(np.linalg.norm(tp - g, axis=1)) > 0.03:
            return
        w, V = np.linalg.eigh(np.cov((near - near.mean(0)).T) + 1e-12 * np.eye(2))
        if w[1] < 3 * w[0]:
            return
        ax = V[:, 1]
        Rg = geom.rotvec_to_R(seg["R_grasp"])
        cands = [i for i in range(3) if i != R.approach_axis]
        def perp(i):
            h = Rg[:2, i]
            n = np.linalg.norm(h)
            return 0 if n < 0.5 else 1 - abs(h @ ax) / n
        best = max(cands, key=perp)
        log(f"closing axis from video tool grasp: {best} (perp scores {[round(perp(i), 2) for i in cands]}), default was {R.closing_axis}")
        if perp(best) > 0.8 and perp(best) - min(perp(i) for i in cands) > 0.5:
            R.closing_axis = best
            R.closing_confirmed = True
    except Exception as e:
        log("infer_closing_axis failed", repr(e))


def reconcile_target(scene, imf, log):
    """Cross-check the detected target with the video's final cube position (same-scene videos only)."""
    try:
        vf = imf.get("cube_final")
        if vf is None or scene.cube is None or imf.get("cube_disp", 0) < 0.04 or imf.get("cube0") is None:
            return
        same = np.linalg.norm(np.array(imf["cube0"]) - np.asarray(scene.cube["center"])[:2]) < 0.025
        log(f"video final cube {np.round(vf, 3).tolist()} same_scene={same}")
        if not same:
            return
        vf = np.array(vf)
        t = scene.target
        if t is None:
            scene.target = {"center": vf, "radius": 0.05, "src": "video"}
            log("target from video final cube position")
            return
        if np.linalg.norm(np.asarray(t["center"]) - vf) < 0.04:
            return
        for c in t.get("cands", []):
            if np.linalg.norm(np.array(c["center"]) - vf) < 0.04:
                log("switching target to candidate matching video", c)
                scene.target = {"center": np.array(c["center"]), "radius": c["radius"], "src": "cand-video"}
                return
        log("WARNING: detected target disagrees with video final cube; keeping detection")
    except Exception as e:
        log("reconcile_target failed", repr(e))


def survey_high(R, scene, log):
    """Escalation: raise the arm (front view less occluded) and re-survey with the wrist camera too."""
    try:
        tz = scene.table_z if scene.table_z is not None else R.pos[2] - 0.2
        p = R.pos.copy()
        p[2] = max(p[2], tz + 0.35)
        R.move_to(p, None, -1.0, tol=0.01, max_iter=5, label="survey high")
        scene.survey()
    except (EpisodeEnded, OutOfBudget):
        raise
    except Exception:
        log("survey_high failed:", traceback.format_exc()[-800:])
