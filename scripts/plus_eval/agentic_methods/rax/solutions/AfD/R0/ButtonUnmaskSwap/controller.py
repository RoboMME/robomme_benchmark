"""ButtonUnmaskSwap autonomous controller (RAX).  Entry point: main(ctx)."""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import percep  # noqa: E402
import rax_io  # noqa: E402
from ctrl import EpisodeOver, Log, Runner, make_row  # noqa: E402
from task import Task  # noqa: E402

VERSION = "bums-v3"


def _limit_from(meta, default=480):
    for key in ["step_limit", "max_steps", "max_episode_steps", "episode_step_limit", "horizon", "max_step"]:
        v = rax_io.find_key(meta, [key])
        try:
            if v is not None and not isinstance(v, (dict, list)) and 20 < int(v) < 100000:
                return int(v)
        except Exception:
            pass
    return default


def video_obs(frame):
    """Build an Obs from a conditioning-video manifest frame (files relative to /workspace)."""
    o = rax_io.Obs()
    o.state = np.asarray(frame.get("state"), dtype=np.float64) if frame.get("state") is not None else None
    for e in frame.get("files", []):
        cam = str(e.get("camera", "")).lower()
        kind = str(e.get("kind", "")).lower()
        im = rax_io.load_file(e.get("path", ""))
        if im is None:
            continue
        k, arr = rax_io._pil_to_array(im)
        if kind == "depth" and k == "depth16":
            o.depth[cam] = rax_io._to_metres(arr, e.get("meters_per_unit"))
        elif kind == "rgb":
            o.rgb[cam] = np.array(im.convert("RGB"))
    return o


def conditioning_frames(meta):
    mp = rax_io.find_key(meta, ["manifest_path"])
    if not isinstance(mp, str):
        return []
    try:
        p = mp if os.path.isabs(mp) else os.path.join("/workspace", mp)
        man = json.load(open(p))
        fr = [f for f in man.get("frames", []) if f.get("phase") == "video_demo"]
        fr.sort(key=lambda f: f.get("frame_index", 0))
        return fr
    except Exception:
        return []


def run_video_prefix(task, frames, log):
    """If containers already hide the cubes, recover identities from the conditioning video."""
    try:
        if not frames:
            return False
        o0 = video_obs(frames[0])
        if not o0.rgb:
            return False
        saved_state = task.r.obs.state
        if o0.state is None:
            o0.state = saved_state
        o0.meta = {}
        cur = task.r.obs
        task.r.obs = o0
        task.detect_cubes(o0)
        ok = all(c in task.cubes and task.cubes[c] for c in task.targets)
        log("video_cubes", ok=ok, n_frames=len(frames))
        if not ok:
            task.r.obs = cur
            return False
        # find the cover frame and track through the rest
        started = False
        for i, f in enumerate(frames[1:], 1):
            o = video_obs(f)
            if o.state is None:
                o.state = saved_state
            task.r.obs = o
            if not started:
                task.cube_px = {c: task.cubes[c][0]["dets"] for c in task.targets}
                vis = [task.colour_visible(o, c) for c in task.targets]
                vis = [v for v in vis if v is not None]
                if vis and max(vis) < 0.25:
                    task.init_tracks(o)
                    started = True
            else:
                task.r.step_count = -len(frames) + i
                task.on_obs(o)
        task.r.obs = cur
        task.r.step_count = cur.step_id or 0
        if started:
            # re-anchor track timestamps to the live clock
            for t in task.tracker.tracks:
                t.last_seen = task.r.step_count
                t.last_step = task.r.step_count
                t.hist = [(task.r.step_count, float(t.pos[0]), float(t.pos[1]))]
        log("video_prefix_done", started=started)
        return started
    except Exception as e:
        log("video_prefix_error", err=repr(e)[:300], tb=traceback.format_exc()[-1200:])
        return False


def main(ctx):
    log = Log()
    log("start", version=VERSION)
    r = Runner(ctx, log)
    task = Task(r, log)
    r.on_obs = task.on_obs
    try:
        st = r.status()
        r.limit = _limit_from(st, 480)
        obs = r.observe()
        if obs.step_id is not None:
            r.step_count = obs.step_id
        if obs.state is None:
            log("no_state", meta_keys=list(obs.meta.keys()) if isinstance(obs.meta, dict) else None)
        ins = obs.instruction or rax_io.find_key(st, ["instruction", "goal"]) or ""
        targets = percep.parse_targets(ins) or ["blue", "green", "red"]
        task.targets = targets
        lim2 = _limit_from(obs.meta, 0)
        if lim2:
            r.limit = lim2
        log("episode", instruction=ins, targets=targets, limit=r.limit, step=r.step_count,
            rgb=list(obs.rgb), depth=list(obs.depth), notes=obs.notes,
            depth_stats={c: [float(d.min()), float(np.median(d)), float(d.max())] for c, d in obs.depth.items()})
        frames = conditioning_frames(obs.meta)
        log("conditioning", n=len(frames))
    except EpisodeOver:
        return
    except Exception as e:
        log("init_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])
        if r.obs is None:
            return

    # ---------------- setup: calibration, cubes, buttons
    try:
        task.setup(r.obs)
    except EpisodeOver:
        return
    except Exception as e:
        log("setup_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])

    # ---------------- drop and tracker init
    try:
        visible = any(c in task.cubes and task.cubes[c] for c in task.targets)
        if not visible and frames:
            if run_video_prefix(task, frames, log):
                visible = True
        if task.tracker is None:
            task.wait_drop()
            r.step([make_row(r, task.home_pos, task.home_R, -1.0, vmax=0.2)])
            task.init_tracks(r.obs)
    except EpisodeOver:
        return
    except Exception as e:
        log("drop_init_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])

    # ---------------- buttons
    try:
        task.press_buttons()
    except EpisodeOver:
        return
    except Exception as e:
        log("press_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])
    try:
        task.go_home()
    except EpisodeOver:
        return
    except Exception as e:
        log("home_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])

    # ---------------- wait for the shuffle to settle
    n_t = max(1, len(task.targets))
    reserve = 55 * n_t + 12
    deadline = r.limit - reserve
    if r.limit <= 600:
        # demo shuffles stopped by ~frame 265; beyond that assume a continuous shuffle and start picking
        deadline = min(deadline, 285)
    deadline = max(r.step_count + 5, deadline)
    try:
        if task.tracker is not None:
            task.wait_static(deadline)
    except EpisodeOver:
        return
    except Exception as e:
        log("wait_error", err=repr(e)[:300], tb=traceback.format_exc()[-2000:])

    # ---------------- picks (strictly in the instructed order; never skip a target)
    for i, c in enumerate(task.targets):
        ok = False
        for tries in range(2):
            try:
                ok = task.pick(c, last=(i == len(task.targets) - 1))
                log("pick_result", color=c, ok=ok, step=r.step_count, tries=tries)
                break
            except EpisodeOver:
                return
            except Exception as e:
                log("pick_error", color=c, err=repr(e)[:300], tb=traceback.format_exc()[-2000:])
                try:
                    if task.held_track is not None:
                        task.gripper_to(-1.0, None, max_steps=5)
                        task.held_track.held = False
                        task.held_track = None
                except EpisodeOver:
                    return
                except Exception:
                    pass
        if not ok:
            # lifting a later target first would break the required order: stop here
            log("pick_abort_order", color=c)
            break
    # ---------------- hold until the episode ends
    try:
        hold_target = r.pos.copy()
        n = 0
        while not r.ended and r.step_count < r.limit and n < 40:
            r.step([make_row(r, hold_target, None, +1.0, vmax=0.2)])
            n += 1
    except EpisodeOver:
        pass
    except Exception as e:
        log("final_hold_error", err=repr(e)[:300])
    log("done", step=r.step_count, ended=r.ended, success=r.success, picked=task.picked,
        elapsed=round(r.elapsed(), 1), avg_step=r.avg_step_time())
