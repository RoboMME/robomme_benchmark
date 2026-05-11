"""Single env × single episode evaluator (val split).

Wrapper around `challenge_interface.scripts.phase1_eval.run_episode` used by the
Claude-driven "Val Seed Model Evaluate Pipeline" (see CLAUDE.md). Defaults to
the 1 task × 1 seed budget (VideoUnmaskSwap ep 0); a full 16-env smoke is
phase1_eval.py instead.
"""
import argparse
import os
import imageio

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from challenge_interface.client import PolicyClient
from challenge_interface.client_http import PolicyHTTPClient
from challenge_interface.scripts.phase1_eval import (
    EXPECTED_ACTION_SHAPES,
    VALID_ACTION_SPACES,
    run_episode,
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
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--team_id", type=str, default="single_smoke",
                        help="Output subdirectory under challenge_results/.")
    return parser.parse_args()


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

    outcome, video_frames, task_goal = run_episode(
        client,
        env_builder,
        args.episode,
        args.env,
        use_depth=args.use_depth,
        use_camera_params=args.use_camera_params,
        action_space=args.action_space,
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
