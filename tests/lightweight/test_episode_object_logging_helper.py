from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.robomme_env.utils.logging import object_log as objectlog


pytestmark = [pytest.mark.lightweight]


class _Pose:
    def __init__(self, position):
        self.p = position


class _Actor:
    def __init__(self, name: str, position):
        self.name = name
        self.pose = _Pose(position)


def _snapshot(actor: _Actor, color: str | None):
    return {
        "name": actor.name,
        "position": pytest.approx(objectlog.extract_actor_world_position(actor)),
        "color": color,
    }


def test_record_object_writes_payload_under_event_key():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    bin_actor = _Actor("bin_0", torch.tensor([[0.2, 0.0, 0.04]], dtype=torch.float32))
    cube_actor = _Actor("target_cube_red", [0.2, 0.0, 0.02])
    target_actor = _Actor("target_cube_green", [0.0, 0.1, 0.02])

    assert objectlog.extract_actor_world_position(bin_actor) == pytest.approx([0.2, 0.0, 0.04])

    objectlog.record_object(
        env,
        event="reset",
        payload={
            "bin_list": [
                {
                    "name": bin_actor.name,
                    "position": objectlog.extract_actor_world_position(bin_actor),
                    "color": "red",
                }
            ],
            "cube_list": [
                {
                    "name": cube_actor.name,
                    "position": objectlog.extract_actor_world_position(cube_actor),
                    "color": "red",
                }
            ],
            "target_cube_list": [
                {
                    "name": target_actor.name,
                    "position": objectlog.extract_actor_world_position(target_actor),
                    "color": "green",
                }
            ],
        },
    )

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=3,
        seed=9,
    )
    assert record["env"] == "DummyEnv"
    assert record["episode"] == 3
    assert record["seed"] == 9
    assert record["object_events"] == {
        "reset": {
            "bin_list": [_snapshot(bin_actor, "red")],
            "cube_list": [_snapshot(cube_actor, "red")],
            "target_cube_list": [_snapshot(target_actor, "green")],
        }
    }
    assert record["swap_events"] == []
    assert record["collision_events"] == []


def test_record_swap_writes_only_swap_index_and_actor_names():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_swap(
        env,
        swap_index=1,
        object_a=_Actor("bin_0", [0.0, 0.0, 0.0]),
        object_b=_Actor("bin_1", [0.1, 0.0, 0.0]),
    )

    record = objectlog.build_episode_object_log_record(
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
    assert record["collision_events"] == []
    assert set(record.keys()) == {
        "env",
        "episode",
        "seed",
        "object_events",
        "swap_events",
        "collision_events",
    }


def test_record_collision_writes_contact_summary_fields():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_collision(
        env,
        contact_summary={
            "swap_contact_detected": True,
            "first_contact_step": 224,
            "contact_pairs": ["bin_1<->bin_2"],
            "max_force_norm": 0.010015421144429798,
            "max_force_pair": "bin_1<->bin_2",
            "max_force_step": 224,
            "pair_max_force": {"bin_1<->bin_2": 0.010015421144429798},
        },
    )

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=5,
        seed=12,
    )
    assert record["collision_events"] == [
        {
            "swap_contact_detected": True,
            "first_contact_step": 224,
            "contact_pairs": ["bin_1<->bin_2"],
            "max_force_norm": 0.010015421144429798,
            "max_force_pair": "bin_1<->bin_2",
            "max_force_step": 224,
            "pair_max_force": {"bin_1<->bin_2": 0.010015421144429798},
        }
    ]


def test_record_object_same_event_overwrites_previous_payload():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_object(
        env,
        event="note",
        payload={"value": 1, "items": ["old"]},
    )
    objectlog.record_object(
        env,
        event="note",
        payload={"value": 2, "items": ["new"]},
    )

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=5,
        seed=12,
    )
    assert record["object_events"] == {
        "note": {"value": 2, "items": ["new"]},
    }


def test_record_object_supports_multiple_events():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_object(env, event="note", payload={"value": 1})
    objectlog.record_object(env, event="snapshot", payload={"count": 3})

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=5,
        seed=12,
    )
    assert record["object_events"] == {
        "note": {"value": 1},
        "snapshot": {"count": 3},
    }


def test_record_object_boundary_event_resets_previous_swap_and_collision_events():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_swap(
        env,
        swap_index=1,
        object_a=_Actor("bin_0", [0.0, 0.0, 0.0]),
        object_b=_Actor("bin_1", [0.1, 0.0, 0.0]),
    )
    objectlog.record_collision(
        env,
        contact_summary={
            "swap_contact_detected": True,
            "first_contact_step": 10,
        },
    )
    objectlog.record_object(env, event="note", payload={"value": "old"})

    objectlog.record_object(
        env,
        event="reset",
        payload={
            "bin_list": [
                {
                    "name": "bin_2",
                    "position": [0.2, 0.0, 0.04],
                    "color": "blue",
                }
            ],
            "cube_list": [
                {
                    "name": "cube_blue",
                    "position": [0.2, 0.0, 0.02],
                    "color": "blue",
                }
            ],
            "target_cube_list": [],
        },
    )

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=6,
        seed=13,
    )
    assert record["object_events"] == {
        "reset": {
            "bin_list": [
                {
                    "name": "bin_2",
                    "position": pytest.approx([0.2, 0.0, 0.04]),
                    "color": "blue",
                }
            ],
            "cube_list": [
                {
                    "name": "cube_blue",
                    "position": pytest.approx([0.2, 0.0, 0.02]),
                    "color": "blue",
                }
            ],
            "target_cube_list": [],
        }
    }
    assert record["swap_events"] == []
    assert record["collision_events"] == []


def test_record_object_init_resets_previous_object_events():
    env = SimpleNamespace()
    objectlog.init_episode_log(env)

    objectlog.record_object(env, event="snapshot", payload={"value": "old"})
    objectlog.record_object(env, event="init", payload={"value": "new"})

    record = objectlog.build_episode_object_log_record(
        env,
        env_id="DummyEnv",
        episode=6,
        seed=13,
    )
    assert record["object_events"] == {"init": {"value": "new"}}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
