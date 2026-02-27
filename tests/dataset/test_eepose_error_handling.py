# -*- coding: utf-8 -*-
"""
test_eepose_error_handling.py
=============================
Heavy test: Verifies that in the `ee_pose` action space, sending an unreachable
target position to the environment properly triggers an error caught by the 
DemonstrationWrapper, returning info["status"] = "error" instead of crashing.

This test iterates over all 16 tasks defined in the benchmark.

Run with:
    cd /data/hongzefu/robomme_benchmark
    uv run python -m pytest tests/dataset/test_eepose_error_handling.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests._shared.repo_paths import find_repo_root

# Ensure the src directory is on the path
pytestmark = pytest.mark.dataset

_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.robomme_env import *  # noqa: F401,F403

# All 16 benchmark tasks
ALL_TASKS = [
    "BinFill", 
    "PickXtimes", 
    "SwingXtimes", 
    "StopCube",
    "VideoUnmask", 
    "VideoUnmaskSwap", 
    "ButtonUnmask", 
    "ButtonUnmaskSwap",
    "PickHighlight", 
    "VideoRepick", 
    "VideoPlaceButton", 
    "VideoPlaceOrder",
    "MoveCube", 
    "InsertPeg", 
    "PatternLock", 
    "RouteStick"
]


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_eepose_unreachable_action_status_error(task_id: str) -> None:
    """
    Instantiate the environment for the given task with 'ee_pose' action space.
    Send a completely unreachable position to provoke an IK/physics failure.
    Assert that DemonstrationWrapper catches it and sets info["status"] = "error".
    """
    print(f"\n[{task_id}] Building environment for ee_pose error handling test...")
    
    env_builder = BenchmarkEnvBuilder(
        env_id=task_id,
        dataset="train",
        action_space="ee_pose",
        gui_render=False,
    )
    
    # We don't need all the observations for this test, keeping it lightweight
    env = env_builder.make_env_for_episode(
        episode_idx=0,
        include_maniskill_obs=False,
    )
    
    try:
        # 1. Reset the environment
        env.reset()
        
        # 2. Define an unreachable action
        # The ee_pose space is (x, y, z, r, p, y, gripper)
        # Coordinates (100.0, 100.0, 100.0) are far out of reach for the robot.
        unreachable_action = [100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 1.0]

        # 3. Step the environment
        obs, reward, terminated, truncated, info = env.step(unreachable_action)

        # 4. Assertions
        assert info is not None, "info dict should not be None"
        
        status = info.get("status")
        assert status == "error", f"Expected info['status'] == 'error', got {status!r}"
        
        error_msg = info.get("error_message")
        assert error_msg is not None, "Expected 'error_message' to be present in info"
        assert isinstance(error_msg, str) and len(error_msg) > 0, (
            f"Expected 'error_message' to be a non-empty string, got {error_msg!r}"
        )

        print(f"[{task_id}] ✓ Successfully caught error: {error_msg}")

    finally:
        env.close()


def main() -> None:
    print("Run this test with pytest:")
    print("uv run python -m pytest tests/dataset/test_eepose_error_handling.py -v -s")
    sys.exit(2)


if __name__ == "__main__":
    main()
