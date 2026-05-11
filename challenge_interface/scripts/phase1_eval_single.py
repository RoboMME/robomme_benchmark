"""Single env × single episode evaluator (val split).

Wrapper around `challenge_interface.scripts.phase1_eval` for the
Claude-driven "Val Seed Model Evaluate Pipeline" (see CLAUDE.md). Defaults to
VideoUnmaskSwap ep 0; a full 16-env smoke is phase1_eval.py instead.

Unlike phase1_eval.run_episode, this script caps env.step calls after reset
via MAX_CLIENT_STEPS — exceeding the cap forces outcome="timeout". This
protects against episodes where the model cannot make demo-phase progress
and would otherwise stream frames until the server's mem_buffer pos_emb
budget is exhausted (see RoboMME/robomme_policy_learning#3).
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


# env.step calls after reset; excludes the video/demonstration phase batched by reset()
MAX_CLIENT_STEPS = 4096


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
    parser.add_argument("--max_client_steps", type=int, default=MAX_CLIENT_STEPS,
                        help="Client-side cap on env.step after reset; "
                             "hitting it forces outcome=timeout (default: %(default)s).")
    parser.add_argument("--team_id", type=str, default="single_smoke",
                        help="Output subdirectory under challenge_results/.")
    return parser.parse_args()


def _run_episode_with_step_cap(
    client,
    env_builder,
    episode_idx,
    env_id,
    *,
    use_depth: bool,
    use_camera_params: bool,
    action_space: str,
    max_client_steps: int,
):
    """Like phase1_eval.run_episode, but cap env.step count after reset."""
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
    action_plan = collections.deque()
    obs, info = env.reset()
    inputs = _build_inputs(obs, info, use_camera_params)
    expected_shape = EXPECTED_ACTION_SHAPES[action_space]

    video_frames = []
    exec_start_idx = len(obs["front_rgb_list"]) - 1

    for i in range(len(obs["front_rgb_list"])):
        video_frames.append(np.hstack([obs["front_rgb_list"][i], obs["wrist_rgb_list"][i]]))
        if i < exec_start_idx:
            video_frames[-1] = cv2.rectangle(
                video_frames[-1], (0, 0),
                (video_frames[-1].shape[1], video_frames[-1].shape[0]),
                (255, 0, 0), 10,
            )

    client_step = 0
    while True:
        if not action_plan:
            outputs = client.infer(inputs)
            action_chunk = outputs["actions"]
            action_plan.extend(action_chunk)
            _clear_inputs(inputs, obs)

        action = action_plan.popleft()
        assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"

        obs, _, terminated, truncated, info = env.step(action)
        video_frames.append(np.hstack([obs["front_rgb_list"][-1], obs["wrist_rgb_list"][-1]]))
        _update_inputs(inputs, obs)

        client_step += 1
        if terminated or truncated:
            break
        if client_step >= max_client_steps:
            info["status"] = "timeout"
            print(f"[CAP] hit max_client_steps={max_client_steps}, forcing outcome=timeout")
            break

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

    if args.transport == "http":
        client = PolicyHTTPClient(host=args.host, port=args.port)
    else:
        client = PolicyClient(host=args.host, port=args.port)

    env_builder = BenchmarkEnvBuilder(
        env_id=args.env,
        dataset="val",
        action_space=args.action_space,
        max_steps=args.max_steps,
    )

    outcome, video_frames, task_goal = _run_episode_with_step_cap(
        client,
        env_builder,
        args.episode,
        args.env,
        use_depth=args.use_depth,
        use_camera_params=args.use_camera_params,
        action_space=args.action_space,
        max_client_steps=args.max_client_steps,
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
