# -*- coding: utf-8 -*-
"""
test_obs_config.py
===================
Integration test: verify that make_env_for_episode include_* flags
correctly control which obs/info fields are present in reset() and step() output.

Tests:
  1. Default (all True):  all 8 optional fields present in obs/info
  2. All disabled (all False): none of the 8 optional fields present
  3. Selective: only front_depth enabled, others False -> only front_depth present
  4. Always-present fields unaffected by any flag combination

Run with:
    cd /data/hongzefu/robomme_benchmark
    uv run python -m pytest tests/dataset/test_obs_config.py -v -s
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.robomme_env import *  # noqa: F401,F403
from robomme.robomme_env.utils import *  # noqa: F401,F403
from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
TEST_ENV_ID = "VideoUnmaskSwap"
TEST_EPISODE = 0
MAX_STEPS_ENV = 1000

# The 8 optional obs fields and where they live
OBS_OPTIONAL_FIELDS = [
    "maniskill_obs",
    "front_depth_list",
    "wrist_depth_list",
    "front_camera_extrinsic_list",
    "wrist_camera_extrinsic_list",
]
INFO_OPTIONAL_FIELDS = [
    "available_multi_choices",
    "front_camera_intrinsic",
    "wrist_camera_intrinsic",
]

# Fields that must ALWAYS be present regardless of flags
OBS_ALWAYS_FIELDS = [
    "front_rgb_list",
    "wrist_rgb_list",
    "joint_state_list",
    "eef_state_list",
    "gripper_state_list",
]
INFO_ALWAYS_FIELDS = [
    "simple_subgoal_online",
    "grounded_subgoal_online",
    "task_goal",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_env(
    include_maniskill_obs=True,
    include_front_depth=True,
    include_wrist_depth=True,
    include_front_camera_extrinsic=True,
    include_wrist_camera_extrinsic=True,
    include_available_multi_choices=True,
    include_front_camera_intrinsic=True,
    include_wrist_camera_intrinsic=True,
):
    builder = BenchmarkEnvBuilder(
        env_id=TEST_ENV_ID,
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
    )
    return builder.make_env_for_episode(
        TEST_EPISODE,
        max_steps=MAX_STEPS_ENV,
        include_maniskill_obs=include_maniskill_obs,
        include_front_depth=include_front_depth,
        include_wrist_depth=include_wrist_depth,
        include_front_camera_extrinsic=include_front_camera_extrinsic,
        include_wrist_camera_extrinsic=include_wrist_camera_extrinsic,
        include_available_multi_choices=include_available_multi_choices,
        include_front_camera_intrinsic=include_front_camera_intrinsic,
        include_wrist_camera_intrinsic=include_wrist_camera_intrinsic,
    )


def _get_first_step_action():
    """Return a simple no-op joint action for testing."""
    return np.zeros(8, dtype=np.float64)


def _check_always_present(obs, info, tag):
    """Assert always-present fields are in obs and info."""
    for field in OBS_ALWAYS_FIELDS:
        assert field in obs, f"[{tag}] always-present obs field '{field}' is missing"
        lst = obs[field]
        assert isinstance(lst, list) and len(lst) > 0, (
            f"[{tag}] obs['{field}'] should be non-empty list, got {type(lst)}"
        )
    for field in INFO_ALWAYS_FIELDS:
        assert field in info, f"[{tag}] always-present info field '{field}' is missing"


def _check_optional_present(obs, info, tag):
    """Assert all 8 optional fields are present."""
    for field in OBS_OPTIONAL_FIELDS:
        assert field in obs, f"[{tag}] optional obs field '{field}' should be present but missing"
    for field in INFO_OPTIONAL_FIELDS:
        assert field in info, f"[{tag}] optional info field '{field}' should be present but missing"


def _check_optional_absent(obs, info, tag):
    """Assert all 8 optional fields are absent."""
    for field in OBS_OPTIONAL_FIELDS:
        assert field not in obs, f"[{tag}] optional obs field '{field}' should be absent but is present"
    for field in INFO_OPTIONAL_FIELDS:
        assert field not in info, f"[{tag}] optional info field '{field}' should be absent but is present"


# ──────────────────────────────────────────────────────────────────────────────
# Test cases
# ──────────────────────────────────────────────────────────────────────────────

def test_all_included(video_unmaskswap_train_ep0_dataset):
    """Default: all flags True -> all 8 optional fields present."""
    print("\n[TEST 1] All flags True (default behavior)")
    env = _make_env()  # all True by default
    resolver = EpisodeDatasetResolver(
        env_id=TEST_ENV_ID,
        episode=TEST_EPISODE,
        dataset_directory=str(video_unmaskswap_train_ep0_dataset.resolver_dataset_dir),
    )
    try:
        obs, info = env.reset()
        _check_always_present(obs, info, "reset/all-included")
        _check_optional_present(obs, info, "reset/all-included")
        print("  RESET: all optional fields present ✓")

        action = resolver.get_step("joint_angle", 0)
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            _check_always_present(obs, info, "step/all-included")
            _check_optional_present(obs, info, "step/all-included")
            print("  STEP: all optional fields present ✓")

        # Spot-check dtypes of optional fields from last obs/info
        _check_optional_dtypes(obs, info, "all-included")
    finally:
        env.close()
    print("  [TEST 1] PASS")


def _check_optional_dtypes(obs, info, tag):
    """Spot-check dtypes of optional fields when present."""
    if "front_depth_list" in obs:
        item = obs["front_depth_list"][-1]
        assert isinstance(item, np.ndarray) and item.dtype == np.int16, (
            f"[{tag}] front_depth_list dtype={item.dtype}, expected int16"
        )
    if "wrist_depth_list" in obs:
        item = obs["wrist_depth_list"][-1]
        assert isinstance(item, np.ndarray) and item.dtype == np.int16, (
            f"[{tag}] wrist_depth_list dtype={item.dtype}, expected int16"
        )
    if "front_camera_extrinsic_list" in obs:
        item = obs["front_camera_extrinsic_list"][-1]
        assert isinstance(item, np.ndarray) and item.dtype == np.float32 and item.shape == (3, 4), (
            f"[{tag}] front_camera_extrinsic_list shape={item.shape} dtype={item.dtype}"
        )
    if "wrist_camera_extrinsic_list" in obs:
        item = obs["wrist_camera_extrinsic_list"][-1]
        assert isinstance(item, np.ndarray) and item.dtype == np.float32 and item.shape == (3, 4), (
            f"[{tag}] wrist_camera_extrinsic_list shape={item.shape} dtype={item.dtype}"
        )
    if "front_camera_intrinsic" in info:
        item = info["front_camera_intrinsic"]
        assert isinstance(item, np.ndarray) and item.dtype == np.float32 and item.shape == (3, 3), (
            f"[{tag}] front_camera_intrinsic shape={item.shape} dtype={item.dtype}"
        )
    if "wrist_camera_intrinsic" in info:
        item = info["wrist_camera_intrinsic"]
        assert isinstance(item, np.ndarray) and item.dtype == np.float32 and item.shape == (3, 3), (
            f"[{tag}] wrist_camera_intrinsic shape={item.shape} dtype={item.dtype}"
        )
    if "available_multi_choices" in info:
        choices = info["available_multi_choices"]
        assert isinstance(choices, list), (
            f"[{tag}] available_multi_choices expected list, got {type(choices)}"
        )


def test_all_excluded(video_unmaskswap_train_ep0_dataset):
    """All flags False -> none of the 8 optional fields present; always-present fields still there."""
    print("\n[TEST 2] All flags False")
    env = _make_env(
        include_maniskill_obs=False,
        include_front_depth=False,
        include_wrist_depth=False,
        include_front_camera_extrinsic=False,
        include_wrist_camera_extrinsic=False,
        include_available_multi_choices=False,
        include_front_camera_intrinsic=False,
        include_wrist_camera_intrinsic=False,
    )
    resolver = EpisodeDatasetResolver(
        env_id=TEST_ENV_ID,
        episode=TEST_EPISODE,
        dataset_directory=str(video_unmaskswap_train_ep0_dataset.resolver_dataset_dir),
    )
    try:
        obs, info = env.reset()
        _check_always_present(obs, info, "reset/all-excluded")
        _check_optional_absent(obs, info, "reset/all-excluded")
        print("  RESET: all optional fields absent, always-present fields ok ✓")

        action = resolver.get_step("joint_angle", 0)
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            _check_always_present(obs, info, "step/all-excluded")
            _check_optional_absent(obs, info, "step/all-excluded")
            print("  STEP: all optional fields absent, always-present fields ok ✓")
    finally:
        env.close()
    print("  [TEST 2] PASS")


def test_selective_front_depth_only(video_unmaskswap_train_ep0_dataset):
    """Only front_depth enabled; others disabled."""
    print("\n[TEST 3] Only include_front_depth=True, others False")
    env = _make_env(
        include_maniskill_obs=False,
        include_front_depth=True,
        include_wrist_depth=False,
        include_front_camera_extrinsic=False,
        include_wrist_camera_extrinsic=False,
        include_available_multi_choices=False,
        include_front_camera_intrinsic=False,
        include_wrist_camera_intrinsic=False,
    )
    resolver = EpisodeDatasetResolver(
        env_id=TEST_ENV_ID,
        episode=TEST_EPISODE,
        dataset_directory=str(video_unmaskswap_train_ep0_dataset.resolver_dataset_dir),
    )
    try:
        obs, info = env.reset()
        _check_always_present(obs, info, "reset/selective")
        # front_depth should be present
        assert "front_depth_list" in obs, "front_depth_list should be present"
        item = obs["front_depth_list"][-1]
        assert isinstance(item, np.ndarray) and item.dtype == np.int16, (
            f"front_depth_list dtype={item.dtype}, expected int16"
        )
        # all others should be absent
        for field in ["maniskill_obs", "wrist_depth_list", "front_camera_extrinsic_list", "wrist_camera_extrinsic_list"]:
            assert field not in obs, f"obs['{field}'] should be absent"
        for field in INFO_OPTIONAL_FIELDS:
            assert field not in info, f"info['{field}'] should be absent"
        print("  RESET: front_depth present, others absent ✓")

        action = resolver.get_step("joint_angle", 0)
        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            _check_always_present(obs, info, "step/selective")
            assert "front_depth_list" in obs, "front_depth_list should be present in step"
            for field in ["maniskill_obs", "wrist_depth_list", "front_camera_extrinsic_list", "wrist_camera_extrinsic_list"]:
                assert field not in obs, f"obs['{field}'] should be absent in step"
            for field in INFO_OPTIONAL_FIELDS:
                assert field not in info, f"info['{field}'] should be absent in step"
            print("  STEP: front_depth present, others absent ✓")
    finally:
        env.close()
    print("  [TEST 3] PASS")


def test_always_present_unaffected():
    """Always-present fields appear regardless of which flags are set."""
    print("\n[TEST 4] Always-present fields unaffected by flag combinations")
    for flags in [
        dict(include_maniskill_obs=True, include_front_depth=True, include_wrist_depth=True,
             include_front_camera_extrinsic=True, include_wrist_camera_extrinsic=True,
             include_available_multi_choices=True, include_front_camera_intrinsic=True,
             include_wrist_camera_intrinsic=True),
        dict(include_maniskill_obs=False, include_front_depth=False, include_wrist_depth=False,
             include_front_camera_extrinsic=False, include_wrist_camera_extrinsic=False,
             include_available_multi_choices=False, include_front_camera_intrinsic=False,
             include_wrist_camera_intrinsic=False),
    ]:
        flag_desc = "all-true" if flags["include_maniskill_obs"] else "all-false"
        env = _make_env(**flags)
        try:
            obs, info = env.reset()
            _check_always_present(obs, info, f"reset/{flag_desc}")
            print(f"  RESET [{flag_desc}]: always-present fields ok ✓")
        finally:
            env.close()
    print("  [TEST 4] PASS")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("all_included", test_all_included),
    ("all_excluded", test_all_excluded),
    ("selective_front_depth_only", test_selective_front_depth_only),
    ("always_present_unaffected", test_always_present_unaffected),
]


def main():
    print("test_obs_config main() now relies on pytest fixture-generated dataset.")
    print("Run with: uv run python -m pytest tests/dataset/test_obs_config.py -v -s")
    sys.exit(2)


if __name__ == "__main__":
    main()
