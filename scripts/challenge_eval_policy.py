"""
Sample script to evaluate your remote policy for the challenge.
"""
import collections
import time

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


# Episode config
ENV_ID = "VideoPlaceButton"
DATASET = "test"
EPISODE_IDX = 0
MAX_STEPS = 200  # 1500 for full challenge

VALID_ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint")
EXPECTED_ACTION_SHAPES = {
    "joint_angle": (8,),
    "ee_pose": (7,),
    "waypoint": (7,),
}


def _build_obs_buffer(obs, task_goal):
    """Build the observation buffer sent to the remote policy server."""
    buffer = {
        "task_goal": task_goal,
        "is_first_step": True,
    }
    for key in obs:
        buffer[key] = obs[key]
    return buffer


def _update_buffer(buffer, obs):
    """Append new observation data into the buffer."""
    for key in obs:
        buffer[key].extend(obs[key])


def _clear_buffer_observations(buffer, obs):
    """Clear observation arrays in the buffer (keep task_goal, is_first_step)."""
    buffer["is_first_step"] = False
    for key in obs:
        buffer[key].clear()


def run_episode(client, env_builder):
    """Run one episode: reset env, stream obs to policy, step until done."""
    env = env_builder.make_env_for_episode(
        episode_idx=EPISODE_IDX,
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
    buffer = _build_obs_buffer(obs, task_goal)
    expected_shape = EXPECTED_ACTION_SHAPES[ACTION_SPACE]

    while True:
        if not action_plan:
            action_chunk = client.infer(buffer)["action"]
            action_plan.extend(action_chunk)
            _clear_buffer_observations(buffer, obs)

        action = action_plan.popleft()
        assert action.shape == expected_shape, f"Expected {expected_shape}, got {action.shape}"

        obs, _, terminated, truncated, info = env.step(action)
        _update_buffer(buffer, obs)

        if terminated or truncated:
            break

    outcome = info.get("status", "unknown")
    env.close()
    return outcome


def main():
    assert ACTION_SPACE in VALID_ACTION_SPACES, (
        f"ACTION_SPACE must be one of {VALID_ACTION_SPACES}"
    )

    client = PolicyClient(host=HOST, port=PORT)
    resp = client.reset()
    while not resp.get("reset_finished", False):
        time.sleep(0.1)
    print("Reset finished")

    env_builder = BenchmarkEnvBuilder(
        env_id=ENV_ID,
        dataset=DATASET,
        action_space=ACTION_SPACE,
        gui_render=False,
        max_steps=MAX_STEPS,
    )

    print("Start running")
    outcome = run_episode(client, env_builder)
    print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()
