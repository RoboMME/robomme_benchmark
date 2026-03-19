"""
Sample script to evaluate participant's remote policy for the challenge.

This is used by RoboMME challenge organizers to evaluate the policy for Phase 1.

"""
import collections
import os
import time
import imageio
import cv2
import numpy as np
import argparse
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from challenge_interface.client import PolicyClient




# Participant parameters (you will need to submit this parameters at eval.ai)
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a policy for the CVPR challenge.")
    parser.add_argument("--action_space", type=str, default="joint_angle", help="Action space to use.")
    parser.add_argument(
        "--use_depth",
        action="store_true",
        help="Whether to use depth images.",
    )
    parser.add_argument(
        "--use_camera_params",
        action="store_true",
        help="Whether to use camera parameters.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host/IP to connect to the policy server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to connect to the policy server.")
    parser.add_argument("--team_id", type=str, default="team_0000", help="Team ID.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Maximum number of steps per episode. We set 1500 for RoboMME Challenge.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate. We will use 10 for RoboMME Challenge Phase 1 evaluation")
    return parser.parse_args()


VALID_ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint")
EXPECTED_ACTION_SHAPES = {
    "joint_angle": (8,),
    "ee_pose": (7,),
    "waypoint": (7,),
}


def _build_inputs(obs, info, use_camera_params):
    """Build the observation buffer sent to the remote policy server."""
    buffer = {
        "task_goal": info["task_goal"],
        "is_first_step": True,
    }
    for key in obs:
        buffer[key] = obs[key]
        
    if use_camera_params:
        buffer["front_camera_intrinsic"] = info["front_camera_intrinsic"]
        buffer["wrist_camera_intrinsic"] = info["wrist_camera_intrinsic"]
    return buffer


def _update_inputs(buffer, obs):
    """Append new observation data into the buffer."""
    for key in obs:
        buffer[key].extend(obs[key])


def _clear_inputs(buffer, obs):
    """Clear observation arrays in the buffer (keep task_goal, is_first_step)."""
    buffer["is_first_step"] = False
    for key in obs:
        buffer[key].clear()


def run_episode(
    client,
    env_builder,
    episode_idx,
    env_id,
    *,
    use_depth: bool,
    use_camera_params: bool,
    action_space: str,
    team_id: str,
):
    """Run one episode: reset env, stream obs to policy, step until done."""
    
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
        if i < exec_start_idx: # add red border to indicate the conditioned video frames
            video_frames[-1] = cv2.rectangle(video_frames[-1], (0, 0), (video_frames[-1].shape[1], video_frames[-1].shape[0]), (255, 0, 0), 10)

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

        if terminated or truncated:
            break

    outcome = info.get("status", "unknown")
    env.close()
    del env
    VIDEO_OUTPUT_DIR = f"challenge_results/{team_id}"
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    imageio.mimsave(f"{VIDEO_OUTPUT_DIR}/{env_id}_ep_{episode_idx}_{outcome}.mp4", video_frames, fps=30)
    return outcome


def main() -> None:
    args = parse_args()
    assert args.action_space in VALID_ACTION_SPACES, (
        f"ACTION_SPACE must be one of {VALID_ACTION_SPACES}"
    )
    client = PolicyClient(host=args.host, port=args.port)
    
    for env_id in BenchmarkEnvBuilder.get_task_list():
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space=args.action_space,
            max_steps=args.max_steps,
        )
        
        for episode_idx in range(args.num_episodes):
            outcome = run_episode(
                client,
                env_builder,
                episode_idx,
                env_id,
                use_depth=args.use_depth,
                use_camera_params=args.use_camera_params,
                action_space=args.action_space,
                team_id=args.team_id,
            )
            print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()
