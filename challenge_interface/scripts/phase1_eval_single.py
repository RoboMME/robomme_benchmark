"""Single env x single episode evaluator (val split).

Wrapper around `challenge_interface.scripts.phase1_eval` for the
Claude-driven "Val Seed Model Evaluate Pipeline" (see CLAUDE.md). Defaults to
VideoUnmaskSwap ep 0; a full 16-env smoke is phase1_eval.py instead.

The non-demo step cap lives in DemonstrationWrapper and is controlled via
`--max_steps` (only policy-phase env.step calls are counted; the reset-time
demonstration video frames are excluded by the wrapper itself). When the cap
trips, the wrapper sets `info["status"] = "timeout"`.

Supports `--dummy_action` (skip policy server, emit zero actions every step)
for sanity-checking that env-side cap without a live model.
"""
import argparse
import collections
import os
import time

import cv2
import imageio
import numpy as np

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from challenge_interface.client import PolicyClient
from challenge_interface.client_http import PolicyHTTPClient
from challenge_interface.scripts.phase1_eval import (
    EXPECTED_ACTION_SHAPES,
    VALID_ACTION_SPACES,
    _build_inputs,
    _clear_inputs,
    _update_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single (env, episode) pair on the val split."
    )
    parser.add_argument("--env", type=str, default="VideoUnmaskSwap",
                        help="env_id (default: %(default)s).")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index in the val metadata (default: %(default)s).")
    parser.add_argument("--transport", type=str, choices=("websocket", "http"),
                        default="websocket")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--action_space", type=str, default="joint_angle")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_camera_params", action="store_true")
    parser.add_argument("--max_steps", type=int, default=1500,
                        help="Env wrapper max_steps (truncate cap for non-demo subgoals).")
    parser.add_argument("--dummy_action", action="store_true",
                        help="Skip policy server and emit zero actions every step. "
                             "Useful for verifying the --max_steps cap without a live model.")
    parser.add_argument("--team_id", type=str, default="single_smoke",
                        help="Output subdirectory under challenge_results/.")
    return parser.parse_args()


def _run_episode_single(
    client,
    env_builder,
    episode_idx,
    env_id,
    *,
    use_depth: bool,
    use_camera_params: bool,
    action_space: str,
    dummy_action: bool,
):
    """Run one (env, episode). The non-demo step cap is enforced by DemonstrationWrapper."""
    expected_shape = EXPECTED_ACTION_SHAPES[action_space]

    if not dummy_action:
        resp = client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)
        print(f"Reset finished for policy server, env id: {env_id}, episode idx: {episode_idx}")

    env = env_builder.make_env_for_episode(
        episode_idx=episode_idx,
        include_front_depth=use_depth,
        include_wrist_depth=use_depth,
        include_front_camera_extrinsic=use_camera_params,
        include_wrist_camera_extrinsic=use_camera_params,
        include_front_camera_intrinsic=use_camera_params,
        include_wrist_camera_intrinsic=use_camera_params,
    )
    obs, info = env.reset()

    if dummy_action:
        inputs = None
        action_plan = None
    else:
        inputs = _build_inputs(obs, info, use_camera_params)
        action_plan = collections.deque()

    video_frames = []
    exec_start_idx = len(obs["front_rgb_list"]) - 1
    demo_frames_in_reset = len(obs["front_rgb_list"])
    wrapper_n_after_reset = getattr(env, "steps_without_demonstration", "?")
    wrapper_total_after_reset = getattr(env, "total_steps", "?")
    print(
        f"[INSTR] reset done - demo_frames_in_reset={demo_frames_in_reset}, "
        f"exec_start_idx={exec_start_idx}, "
        f"wrapper.steps_without_demonstration={wrapper_n_after_reset}, "
        f"wrapper.total_steps={wrapper_total_after_reset}",
        flush=True,
    )

    for i in range(len(obs["front_rgb_list"])):
        video_frames.append(np.hstack([obs["front_rgb_list"][i], obs["wrist_rgb_list"][i]]))
        if i < exec_start_idx:
            video_frames[-1] = cv2.rectangle(
                video_frames[-1], (0, 0),
                (video_frames[-1].shape[1], video_frames[-1].shape[0]),
                (255, 0, 0), 10,
            )

    non_demo_step = 0
    while True:
        if dummy_action:
            action = np.zeros(expected_shape, dtype=np.float32)
        else:
            if not action_plan:
                outputs = client.infer(inputs)
                action_chunk = outputs["actions"]
                action_plan.extend(action_chunk)
                _clear_inputs(inputs, obs)
            action = action_plan.popleft()
            assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"

        obs, _, terminated, truncated, info = env.step(action)
        non_demo_step += 1
        video_frames.append(np.hstack([obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1]]))
        if not dummy_action:
            _update_inputs(inputs, obs)

        if non_demo_step % 100 == 0:
            wrapper_n = getattr(env, "steps_without_demonstration", "?")
            wrapper_total = getattr(env, "total_steps", "?")
            print(
                f"[INSTR] non_demo_step={non_demo_step}, "
                f"wrapper.steps_without_demonstration={wrapper_n}, "
                f"wrapper.total_steps={wrapper_total}, "
                f"status={info.get('status')}",
                flush=True,
            )

        if terminated or truncated:
            break

    wrapper_n_final = getattr(env, "steps_without_demonstration", "?")
    wrapper_total_final = getattr(env, "total_steps", "?")
    print(
        f"[INSTR] loop exit - non_demo_step={non_demo_step}, "
        f"wrapper.steps_without_demonstration={wrapper_n_final}, "
        f"wrapper.total_steps={wrapper_total_final}, "
        f"terminated={bool(terminated)}, truncated={bool(truncated)}, "
        f"status={info.get('status')}",
        flush=True,
    )

    outcome = info.get("status", "unknown")
    env.close()
    del env
    return outcome, video_frames, info["task_goal"][0]


def main() -> None:
    args = parse_args()
    assert args.action_space in VALID_ACTION_SPACES, (
        f"action_space must be one of {VALID_ACTION_SPACES}"
    )
    assert args.action_space in EXPECTED_ACTION_SHAPES

    if args.dummy_action:
        client = None
    elif args.transport == "http":
        client = PolicyHTTPClient(host=args.host, port=args.port)
    else:
        client = PolicyClient(host=args.host, port=args.port)

    env_builder = BenchmarkEnvBuilder(
        env_id=args.env,
        dataset="val",
        action_space=args.action_space,
        max_steps=args.max_steps,
    )

    outcome, video_frames, task_goal = _run_episode_single(
        client,
        env_builder,
        args.episode,
        args.env,
        use_depth=args.use_depth,
        use_camera_params=args.use_camera_params,
        action_space=args.action_space,
        dummy_action=args.dummy_action,
    )

    output_dir = os.path.join("challenge_results", args.team_id)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(
        video_dir,
        f"{args.env}_ep_{args.episode}_{outcome}_{task_goal}.mp4",
    )
    imageio.mimsave(video_path, video_frames, fps=30)
    print(f"Outcome: {outcome} (task={args.env}, episode={args.episode})")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
