"""
Sample script to evaluate participant's remote policy for the challenge.

This is used by RoboMME challenge organizers to evaluate the policy.
"""
import collections
import os
import time
import imageio
import cv2
import numpy as np

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from remote_evaluation.client import PolicyClient

# -----------------------------------------------------------------------------
# Participant parameters (submit via JSON on eval.ai)
ACTION_SPACE = "joint_angle"
USE_DEPTH = False
USE_CAMERA_PARAMS = False
HOST = "141.212.115.116"
PORT = 8012
# -----------------------------------------------------------------------------


VALID_ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint")
EXPECTED_ACTION_SHAPES = {
    "joint_angle": (8,),
    "ee_pose": (7,),
    "waypoint": (7,),
}


def _build_inputs(obs, task_goal):
    """Build the observation buffer sent to the remote policy server."""
    buffer = {
        "task_goal": task_goal,
        "is_first_step": True,
    }
    for key in obs:
        buffer[key] = obs[key]
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


def run_episode(client, env_builder, episode_idx, env_id):
    """Run one episode: reset env, stream obs to policy, step until done."""
    
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)
    print(f"Reset finished for policy server, env id: {env_id}, episode idx: {episode_idx}")
    
    env = env_builder.make_env_for_episode(
        episode_idx=episode_idx,
        include_front_depth=USE_DEPTH,
        include_wrist_depth=USE_DEPTH,
        include_front_camera_extrinsic=USE_CAMERA_PARAMS,
        include_wrist_camera_extrinsic=USE_CAMERA_PARAMS,
        include_front_camera_intrinsic=USE_CAMERA_PARAMS,
        include_wrist_camera_intrinsic=USE_CAMERA_PARAMS,
    )
    action_plan = collections.deque()
    obs, info = env.reset()
    task_goal = info["task_goal"]
    inputs = _build_inputs(obs, task_goal)
    expected_shape = EXPECTED_ACTION_SHAPES[ACTION_SPACE]
    
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
    os.makedirs("test_videos", exist_ok=True)
    imageio.mimsave(f"test_videos/{env_id}_ep_{episode_idx}_{outcome}.mp4", video_frames, fps=30)
    return outcome


def main():
    assert ACTION_SPACE in VALID_ACTION_SPACES, (
        f"ACTION_SPACE must be one of {VALID_ACTION_SPACES}"
    )

    client = PolicyClient(host=HOST, port=PORT)
    
    for env_id in BenchmarkEnvBuilder.get_task_list():
        env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space=ACTION_SPACE,
            gui_render=False,
            max_steps=1500,
        )
        
        for episode_idx in range(50):
            outcome = run_episode(client, env_builder, episode_idx, env_id)
            print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()
