from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.env_record_wrapper.episode_object_logging import (
    append_episode_object_swap_event,
    build_episode_object_log_record,
    extract_actor_world_position,
    init_episode_object_log_state,
    record_reset_objects,
)


pytestmark = [pytest.mark.lightweight]


class _Pose:
    def __init__(self, position):
        self.p = position


class _Actor:
    def __init__(self, name: str, position):
        self.name = name
        self.pose = _Pose(position)


def test_record_reset_objects_writes_only_name_position_color():
    env = SimpleNamespace()
    init_episode_object_log_state(env)

    bin_actor = _Actor("bin_0", torch.tensor([[0.2, 0.0, 0.04]], dtype=torch.float32))
    cube_actor = _Actor("target_cube_red", [0.2, 0.0, 0.02])
    target_actor = _Actor("target_cube_green", [0.0, 0.1, 0.02])

    assert extract_actor_world_position(bin_actor) == pytest.approx([0.2, 0.0, 0.04])

    record_reset_objects(
        env,
        bin_list=[{"actor": bin_actor, "color": "red"}],
        cube_list=[{"actor": cube_actor, "color": "red"}],
        target_cube_list=[{"actor": target_actor, "color": "green"}],
    )

    record = build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=3,
        seed=9,
    )
    assert record["env"] == "DummyEnv"
    assert record["episode"] == 3
    assert record["seed"] == 9
    assert record["bin_list"] == [
        {
            "name": "bin_0",
            "position": pytest.approx([0.2, 0.0, 0.04]),
            "color": "red",
        }
    ]
    assert record["cube_list"] == [
        {
            "name": "target_cube_red",
            "position": pytest.approx([0.2, 0.0, 0.02]),
            "color": "red",
        }
    ]
    assert record["target_cube_list"] == [
        {
            "name": "target_cube_green",
            "position": pytest.approx([0.0, 0.1, 0.02]),
            "color": "green",
        }
    ]
    assert record["swap_events"] == []


def test_append_swap_event_writes_only_swap_index_and_actor_names():
    env = SimpleNamespace()
    init_episode_object_log_state(env)

    append_episode_object_swap_event(
        env,
        swap_index=1,
        object_a=_Actor("bin_0", [0.0, 0.0, 0.0]),
        object_b=_Actor("bin_1", [0.1, 0.0, 0.0]),
    )

    record = build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=5,
        seed=12,
    )
    assert record["swap_events"] == [
        {
            "swap_index": 1,
            "object_a": "bin_0",
            "object_b": "bin_1",
        }
    ]
    assert set(record.keys()) == {
        "env",
        "episode",
        "seed",
        "bin_list",
        "cube_list",
        "target_cube_list",
        "swap_events",
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
