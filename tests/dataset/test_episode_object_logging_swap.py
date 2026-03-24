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


def _assert_object_list_shape(items):
    assert isinstance(items, list)
    for item in items:
        assert set(item.keys()) == {"name", "position", "color"}
        assert isinstance(item["name"], (str, type(None)))
        assert isinstance(item["color"], (str, type(None)))
        assert item["position"] is None or len(item["position"]) == 3


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
    assert set(record.keys()) == {
        "env",
        "episode",
        "seed",
        "bin_list",
        "cube_list",
        "target_cube_list",
        "swap_events",
    }
    assert "schema_version" not in record
    assert "difficulty" not in record
    assert "episode_success" not in record
    assert "object_log" not in record

    bin_list = record["bin_list"]
    cube_list = record["cube_list"]
    target_cube_list = record["target_cube_list"]
    assert len(bin_list) == 3
    assert len(cube_list) == 3
    assert len(target_cube_list) == 1
    _assert_object_list_shape(bin_list)
    _assert_object_list_shape(cube_list)
    _assert_object_list_shape(target_cube_list)

    expected_swap_times = _expected_swap_times(
        env_id,
        seed=generated.used_seed,
        difficulty=case.difficulty,
    )
    swap_events = record["swap_events"]
    assert len(swap_events) == expected_swap_times
    assert len({event["swap_index"] for event in swap_events}) == expected_swap_times
    for event in swap_events:
        assert set(event.keys()) == {"swap_index", "object_a", "object_b"}
        assert isinstance(event["swap_index"], int)
        assert isinstance(event["object_a"], (str, type(None)))
        assert isinstance(event["object_b"], (str, type(None)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
