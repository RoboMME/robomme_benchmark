from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.dataset_generation import DatasetCase
from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.env_record_wrapper.episode_object_logging import EPISODE_OBJECT_LOG_FILENAME


pytestmark = pytest.mark.dataset


TEST_CASES = [
    ("ButtonUnmaskSwap", 0),
    ("VideoUnmaskSwap", 0),
]


def _build_case(env_id: str, episode: int) -> DatasetCase:
    builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
    )
    seed, difficulty = builder.resolve_episode(episode)
    return DatasetCase(
        env_id=env_id,
        episode=episode,
        base_seed=int(seed) if seed is not None else 0,
        difficulty=str(difficulty) if difficulty else None,
        save_video=False,
        mode_tag="episode_object_log_swap",
    )


def _load_matching_record(jsonl_path, *, env_id: str, episode: int, seed: int):
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in reversed(records):
        if (
            record.get("env") == env_id
            and int(record.get("episode")) == int(episode)
            and int(record.get("seed")) == int(seed)
        ):
            return record
    raise AssertionError(f"Missing JSONL record for env={env_id} episode={episode} seed={seed}")


def _expected_swap_times(env_id: str, *, seed: int, difficulty: str | None) -> int:
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
    )
    if difficulty:
        env_kwargs["difficulty"] = difficulty
    env = gym.make(env_id, **env_kwargs)
    try:
        return int(env.unwrapped.swap_times)
    finally:
        env.close()


@pytest.mark.parametrize("env_id,episode", TEST_CASES)
def test_swap_env_episode_object_log(dataset_factory, env_id: str, episode: int):
    case = _build_case(env_id, episode)
    generated = dataset_factory(case)

    jsonl_path = generated.work_dir / EPISODE_OBJECT_LOG_FILENAME
    assert jsonl_path.exists(), f"Missing episode object log JSONL: {jsonl_path}"

    record = _load_matching_record(
        jsonl_path,
        env_id=env_id,
        episode=episode,
        seed=generated.used_seed,
    )
    assert record["schema_version"] == 1
    assert record["episode_success"] is True

    object_log = record["object_log"]
    cube_bins = object_log["cube_bins"]
    assert len(cube_bins) == 3
    for entry in cube_bins:
        assert len(entry["bin_world_position"]) == 3
        assert len(entry["cube_world_position"]) == 3
        assert entry["bin"]["type"] == "bin"
        assert entry["cube"]["type"] == "cube"

    target_cube = object_log["target_cube"]
    assert target_cube is not None
    assert target_cube["type"] == "cube"
    assert len(target_cube["world_position"]) == 3

    expected_swap_times = _expected_swap_times(
        env_id,
        seed=generated.used_seed,
        difficulty=case.difficulty,
    )
    swap_events = object_log["swap_events"]
    assert len(swap_events) == expected_swap_times
    assert len({event["swap_index"] for event in swap_events}) == expected_swap_times
    for event in swap_events:
        assert event["object_a"]["type"] == "bin"
        assert event["object_b"]["type"] == "bin"
        assert len(event["object_a"]["world_position"]) == 3
        assert len(event["object_b"]["world_position"]) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
