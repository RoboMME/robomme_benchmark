from __future__ import annotations

import sys

import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.repo_paths import ensure_src_on_path
ensure_src_on_path(__file__)

from robomme.env_record_wrapper.episode_object_logging import (
    build_cube_bin_entry,
    build_object_descriptor,
    empty_episode_object_log,
    extract_actor_world_position,
    normalize_episode_object_log,
)


pytestmark = [pytest.mark.lightweight]


class _Pose:
    def __init__(self, position):
        self.p = position


class _Actor:
    def __init__(self, name: str, position):
        self.name = name
        self.pose = _Pose(position)


def test_object_descriptor_extracts_position_and_metadata():
    actor = _Actor("bin_2", torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32))

    assert extract_actor_world_position(actor) == pytest.approx([0.1, -0.2, 0.3])

    descriptor = build_object_descriptor(
        actor=actor,
        object_type="bin",
        color=None,
        bin_index=2,
    )
    assert descriptor == {
        "type": "bin",
        "name": "bin_2",
        "actor_name": "bin_2",
        "color": None,
        "bin_index": 2,
        "world_position": pytest.approx([0.1, -0.2, 0.3]),
    }


def test_cube_bin_entry_and_normalize_keep_expected_fields():
    bin_actor = _Actor("bin_0", [0.2, 0.0, 0.04])
    cube_actor = _Actor("target_cube_red", [0.2, 0.0, 0.02])

    entry = build_cube_bin_entry(
        bin_actor=bin_actor,
        cube_actor=cube_actor,
        color="red",
        bin_index=0,
    )
    assert entry["bin_index"] == 0
    assert entry["color"] == "red"
    assert entry["bin"]["type"] == "bin"
    assert entry["cube"]["type"] == "cube"
    assert entry["bin_world_position"] == pytest.approx([0.2, 0.0, 0.04])
    assert entry["cube_world_position"] == pytest.approx([0.2, 0.0, 0.02])

    payload = normalize_episode_object_log(
        {
            "cube_bins": [entry],
            "target_cube": build_object_descriptor(
                actor=cube_actor,
                object_type="cube",
                color="red",
                bin_index=0,
            ),
            "swap_events": [
                {
                    "swap_index": 1,
                    "object_a": build_object_descriptor(
                        actor=bin_actor,
                        object_type="bin",
                        bin_index=0,
                    ),
                    "object_b": build_object_descriptor(
                        actor=_Actor("bin_1", [0.0, 0.1, 0.04]),
                        object_type="bin",
                        bin_index=1,
                    ),
                }
            ],
        }
    )
    assert payload["cube_bins"][0]["bin_world_position"] == pytest.approx([0.2, 0.0, 0.04])
    assert payload["target_cube"]["world_position"] == pytest.approx([0.2, 0.0, 0.02])
    assert payload["swap_events"][0]["object_a"]["bin_index"] == 0
    assert normalize_episode_object_log(None) == empty_episode_object_log()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
