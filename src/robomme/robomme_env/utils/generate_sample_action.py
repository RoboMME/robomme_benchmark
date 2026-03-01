from typing import Generator, Optional, Union

import numpy as np

from .constant import (
    EE_POSE_ACTION_SPACE,
    JOINT_ACTION_SPACE,
    WAYPOINT_ACTION_SPACE,
    MULTI_CHOICE_ACTION_SPACE,
)

NOISE_LEVEL = 0.01


def _add_small_noise(
    action: np.ndarray, noise_level: float = 0.0
) -> np.ndarray:
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0  # Preserve gripper action
    return action + noise


def _get_current_joint_action(env) -> np.ndarray:
    """Read current joint positions and gripper state from the env."""
    state = env.unwrapped.agent.robot.qpos
    state_flat = state.cpu().numpy().flatten() if hasattr(state, 'cpu') else np.asarray(state).flatten()
    joint_state = state_flat[:7]  # 7 arm joints
    gripper_state = 1
    return np.concatenate([joint_state, [gripper_state]]).astype(np.float32)


def _get_current_ee_action(env) -> np.ndarray:
    """Read current end-effector pose and gripper state from the env."""
    tcp_pose = env.unwrapped.agent.tcp.pose
    pos = tcp_pose.p.cpu().numpy().flatten() if hasattr(tcp_pose.p, 'cpu') else np.asarray(tcp_pose.p).flatten()
    from robomme.robomme_env.utils.rpy_util import build_endeffector_pose_dict
    ee_dict, _, _ = build_endeffector_pose_dict(tcp_pose.p, tcp_pose.q, None, None)
    rpy = ee_dict['rpy'].cpu().numpy().flatten() if hasattr(ee_dict['rpy'], 'cpu') else np.asarray(ee_dict['rpy']).flatten()
    gripper_state = 1
    return np.concatenate([pos[:3], rpy[:3], [gripper_state]]).astype(np.float32)


def generate_sample_actions(
    action_space: str, env=None, task_id: Optional[str] = None,
) -> Generator[Union[np.ndarray, dict], None, None]:
    if action_space == JOINT_ACTION_SPACE:
        # Read current joint state from env and add small random noise
        while True:
            base = _get_current_joint_action(env)
            yield _add_small_noise(base, noise_level=NOISE_LEVEL)

    elif action_space == EE_POSE_ACTION_SPACE:
        # Read current EE pose from env and add small random noise
        while True:
            base = _get_current_ee_action(env)
            yield _add_small_noise(base, noise_level=NOISE_LEVEL)

    elif action_space == WAYPOINT_ACTION_SPACE:
        # Read current EE pose + gripper; add small noise to xyz only, z-0.1
        while True:
            base = _get_current_ee_action(env)  # [x, y, z, r, p, y, gripper]
            base[:3] += np.random.normal(0, NOISE_LEVEL, 3)
            yield base

    elif action_space == MULTI_CHOICE_ACTION_SPACE:
        # Sample multi-choice actions for demonstration.
        # Format follows dataset convention: uppercase "choice" + optional [y, x] pixel position.
        choices = [
            {"choice": "A", "point": [240, 320]},
            {"choice": "B", "point": [260, 420]},
            {"choice": "C"},
        ]
        for choice in choices:
            yield choice

    else:
        raise ValueError(f"Unsupported action space: {action_space}")
