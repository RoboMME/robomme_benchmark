# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.lightweight


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


select_keyframe = _load_module(
    "select_keyframe_under_test",
    "scripts/dev3/select-keyframe.py",
)


def _make_rgb(fill_value: int) -> np.ndarray:
    return np.full((256, 256, 3), fill_value, dtype=np.uint8)


def _make_timestep(
    episode_group: h5py.Group,
    timestep_name: str,
    *,
    fill_value: int,
    waypoint_action: np.ndarray | None,
    is_video_demo: bool,
) -> None:
    ts = episode_group.create_group(timestep_name)
    obs = ts.create_group("obs")
    obs.create_dataset("front_rgb", data=_make_rgb(fill_value))
    action = ts.create_group("action")
    if waypoint_action is not None:
        action.create_dataset("waypoint_action", data=np.asarray(waypoint_action))
    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)


def test_episode_and_timestep_sorting_and_waypoint_keyframes(tmp_path: Path) -> None:
    h5_path = tmp_path / "record_dataset_Dummy.h5"
    waypoint_a = np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float64)
    waypoint_b = np.array([10, 20, 30, 40, 50, 60, 1], dtype=np.float64)
    waypoint_bad = np.array([np.nan] * 7, dtype=np.float64)

    with h5py.File(h5_path, "w") as h5:
        h5.create_group("episode_10")
        episode = h5.create_group("episode_2")
        h5.create_group("episode_1")

        _make_timestep(
            episode,
            "timestep_3",
            fill_value=3,
            waypoint_action=waypoint_b,
            is_video_demo=False,
        )
        _make_timestep(
            episode,
            "timestep_0",
            fill_value=0,
            waypoint_action=waypoint_a,
            is_video_demo=True,
        )
        _make_timestep(
            episode,
            "timestep_1",
            fill_value=1,
            waypoint_action=waypoint_a,
            is_video_demo=True,
        )
        _make_timestep(
            episode,
            "timestep_3_dup1",
            fill_value=4,
            waypoint_action=waypoint_b,
            is_video_demo=False,
        )
        _make_timestep(
            episode,
            "timestep_2",
            fill_value=2,
            waypoint_action=waypoint_bad,
            is_video_demo=False,
        )
        _make_timestep(
            episode,
            "timestep_4",
            fill_value=5,
            waypoint_action=None,
            is_video_demo=False,
        )
        _make_timestep(
            episode,
            "timestep_5",
            fill_value=6,
            waypoint_action=waypoint_a,
            is_video_demo=False,
        )

    with h5py.File(h5_path, "r") as h5:
        assert select_keyframe.list_episode_indices(h5) == [1, 2, 10]

        episode = h5["episode_2"]
        assert select_keyframe.iter_sorted_timestep_keys(episode) == [
            "timestep_0",
            "timestep_1",
            "timestep_2",
            "timestep_3",
            "timestep_3_dup1",
            "timestep_4",
            "timestep_5",
        ]

        decisions, valid_waypoint_count = select_keyframe.scan_episode_keyframes(episode)

    assert valid_waypoint_count == 5
    assert [decision.timestep_key for decision in decisions if decision.is_keyframe] == [
        "timestep_0",
        "timestep_3",
        "timestep_5",
    ]
    assert [decision.keyframe_index for decision in decisions if decision.is_keyframe] == [1, 2, 3]
    assert decisions[0].is_video_demo is True
    assert decisions[0].is_keyframe is True
    assert decisions[1].is_keyframe is False
    assert decisions[2].waypoint_action is None


def test_render_frame_highlights_only_keyframes() -> None:
    base_rgb = _make_rgb(120)
    non_keyframe = select_keyframe.KeyframeDecision(
        frame_index=1,
        timestep_key="timestep_1",
        is_video_demo=False,
        waypoint_action=None,
        is_keyframe=False,
        keyframe_index=None,
    )
    keyframe = select_keyframe.KeyframeDecision(
        frame_index=2,
        timestep_key="timestep_2",
        is_video_demo=True,
        waypoint_action=np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float64),
        is_keyframe=True,
        keyframe_index=1,
    )

    regular = select_keyframe.render_frame(
        base_rgb,
        task_name="Dummy",
        episode=0,
        decision=non_keyframe,
    )
    highlighted = select_keyframe.render_frame(
        base_rgb,
        task_name="Dummy",
        episode=0,
        decision=keyframe,
    )

    assert regular.shape == (256, 256, 3)
    assert highlighted.shape == (256, 256, 3)
    assert np.array_equal(regular[0, 0], np.array([120, 120, 120], dtype=np.uint8))
    assert np.array_equal(highlighted[0, 0], np.array([0, 0, 255], dtype=np.uint8))
