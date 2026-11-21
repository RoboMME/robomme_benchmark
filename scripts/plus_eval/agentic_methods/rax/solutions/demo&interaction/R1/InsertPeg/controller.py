"""InsertPeg controller (RoboMME-Plus, RAX).

Transition table
----------------
INIT      -> observe, read status/limits, load the reset manifest (conditioning video).
DECIDE    -> if video frames carry usable robot states (moving TCP, gripper closes, start
             matches the execution start): REPLAY, else VISION.
REPLAY    -> track video TCP poses closed-loop; gripper transitions settle at the video's
             grasp pose; after closing, check finger width (one regrasp if empty).
             exit -> PUSH (always), or END on task_complete / episode_ended.
PUSH      -> converge on final video pose, push up to 18 mm further along the insertion
             direction with small wiggles; exit -> END.
VISION    -> depth-based fallback (vision.py): moved peg from the video, held end from wrist
             frames, insertion side from the video's last frame, grasp, align, insert.
END       -> stop sending actions, write summary, return.
Every phase runs inside try/except; any exception moves on to END without crashing.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import rio  # noqa: E402
from rio import log, save_json, safe  # noqa: E402
import replay  # noqa: E402

VERSION = "insertpeg-v1"


def _strip_images(reply):
    out = []
    for b in (reply or {}).get("content", []) or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", "")[:200000])
        elif isinstance(b, dict):
            out.append({"type": b.get("type"), "mimeType": b.get("mimeType"),
                        "len": len(str(b.get("data", "")))})
    return out


def layout_check(video, execf):
    """Mean |depth(video frame 0) - depth(execution frame)| over the front camera, mm."""
    try:
        from PIL import Image
        if not video or not execf:
            return None
        a, _ = replay.file_of(video[0], "front", "depth")
        b, _ = replay.file_of(execf[0], "front", "depth")
        if not a or not b or not os.path.isfile(a) or not os.path.isfile(b):
            return None
        A = np.asarray(Image.open(a)).astype(float)
        B = np.asarray(Image.open(b)).astype(float)
        m = (A > 0) & (B > 0)
        d = np.abs(A - B)[m]
        return float(np.mean(d)), float(np.mean(d > 8.0))
    except Exception as e:
        log("layout check failed", repr(e))
        return None


def main(ctx):
    t0 = time.time()
    log("=== %s start ===" % VERSION)
    robot = rio.Robot(ctx)
    summary = {"version": VERSION}
    try:
        pkt, raw = robot.observe()
        save_json("first_observe_text.json", _strip_images(raw))
        st = robot.status()
        save_json("status.json", st)
        log("state", None if robot.state is None else np.round(robot.state, 4),
            "remaining", robot.remaining, "ended", robot.ended)
        summary["start_state"] = None if robot.state is None else robot.state.tolist()
        summary["remaining_at_start"] = robot.remaining
        try:
            log("instruction:", [v for _, v in rio.find_all(pkt, "instruction")][:1])
        except Exception:
            pass

        manifest, mpath = replay.load_manifest(pkt)
        video, execf = ([], [])
        if manifest is not None:
            video, execf = replay.split_frames(manifest)
        log("manifest", mpath, "video frames", len(video), "exec frames", len(execf))
        summary["manifest"] = mpath
        summary["n_video"] = len(video)
        if video:
            save_json("video_frame0.json", video[0])
            save_json("video_frame_last.json", video[-1])
        lc = layout_check(video, execf)
        log("layout check (mean mm, frac>5mm):", lc)
        summary["layout_check"] = lc

        vs = replay.video_states(video)
        usable, why = replay.states_usable(vs, robot.state)
        log("video states usable:", usable, why)
        summary["replay_usable"] = usable
        summary["replay_why"] = why
        if usable and lc is not None and lc[1] > 0.015:
            log("layout appears changed since the video -> replay would be wrong; use vision")
            usable = False
            summary["replay_usable"] = False
        if robot.state is not None:
            summary["rotation_probe"] = safe(robot.probe_rotation, -1.0 if robot.width() > 0.03 else 1.0)
        if robot.state is None:
            log("no state available; cannot control")
        elif usable:
            summary["method"] = "replay"
            wps = replay.build_waypoints(vs)
            safe(replay.run_replay, robot, vs)
            safe(replay.final_push, robot, wps)
        else:
            summary["method"] = "vision"
            import vision
            safe(vision.run_vision, robot, pkt, video, execf)
    except Exception:
        log("TOP-LEVEL EXCEPTION", traceback.format_exc()[-3000:])
    summary.update({"complete": robot.complete, "ended": robot.ended,
                    "steps_sent": robot.steps_sent, "step_calls": robot.step_calls,
                    "step_time_s": round(robot.step_time, 1),
                    "wall_s": round(time.time() - t0, 1), "rot_mode": robot.rot_mode,
                    "final_state": None if robot.state is None else robot.state.tolist()})
    log("summary", summary)
    save_json("summary.json", summary)
    return summary
