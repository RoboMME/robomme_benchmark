# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.lightweight

_MISSING = object()


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


select_keyframe_jointangle = _load_module(
    "select_keyframe_jointangle_under_test",
    "scripts/dev3/select-keyframe-jointangle.py",
)


def _make_rgb(fill_value: int, side: int = 32) -> np.ndarray:
    return np.full((side, side, 3), fill_value, dtype=np.uint8)


def _make_joint_action(value: float, gripper_value: float = -1.0) -> np.ndarray:
    return np.array([value] * 7 + [gripper_value], dtype=np.float64)


def _make_timestep(
    episode_group: h5py.Group,
    timestep_name: str,
    *,
    fill_value: int,
    joint_action=_MISSING,
    waypoint_action=_MISSING,
    is_gripper_close=_MISSING,
    is_video_demo: bool = False,
) -> None:
    ts = episode_group.create_group(timestep_name)
    obs = ts.create_group("obs")
    obs.create_dataset("front_rgb", data=_make_rgb(fill_value))
    if is_gripper_close is not _MISSING:
        obs.create_dataset("is_gripper_close", data=is_gripper_close)

    action = ts.create_group("action")
    if joint_action is not _MISSING:
        action.create_dataset("joint_action", data=np.asarray(joint_action))
    if waypoint_action is not _MISSING:
        action.create_dataset("waypoint_action", data=np.asarray(waypoint_action))

    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)


def test_scan_episode_keyframes_detects_predicted_and_native_transitions(tmp_path: Path) -> None:
    h5_path = tmp_path / "record_dataset_Dummy.h5"
    waypoint_a = np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float64)
    waypoint_b = np.array([10, 20, 30, 40, 50, 60, 1], dtype=np.float64)
    waypoint_c = np.array([100, 200, 300, 400, 500, 600, -1], dtype=np.float64)
    waypoint_bad = np.full(7, np.nan, dtype=np.float64)
    joint_bad = np.full(8, np.nan, dtype=np.float64)

    with h5py.File(h5_path, "w") as h5:
        h5.create_group("episode_10")
        episode = h5.create_group("episode_2")
        h5.create_group("episode_1")

        _make_timestep(
            episode,
            "timestep_3",
            fill_value=3,
            joint_action=_make_joint_action(0.02),
            waypoint_action=waypoint_b,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_0",
            fill_value=0,
            joint_action=_make_joint_action(0.0),
            waypoint_action=waypoint_a,
            is_gripper_close=False,
            is_video_demo=True,
        )
        _make_timestep(
            episode,
            "timestep_1",
            fill_value=1,
            joint_action=_make_joint_action(0.0),
            waypoint_action=waypoint_a,
            is_gripper_close=False,
            is_video_demo=True,
        )
        _make_timestep(
            episode,
            "timestep_3_dup1",
            fill_value=4,
            joint_action=_make_joint_action(0.02),
            waypoint_action=waypoint_b,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_2",
            fill_value=2,
            joint_action=_make_joint_action(0.01),
            waypoint_action=waypoint_bad,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_4",
            fill_value=5,
            joint_action=_make_joint_action(0.02),
            waypoint_action=_MISSING,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_5",
            fill_value=6,
            joint_action=_make_joint_action(0.03),
            waypoint_action=waypoint_c,
            is_gripper_close=True,
        )
        _make_timestep(
            episode,
            "timestep_6",
            fill_value=7,
            joint_action=joint_bad,
            waypoint_action=waypoint_a,
            is_gripper_close=True,
        )
        _make_timestep(
            episode,
            "timestep_7",
            fill_value=8,
            joint_action=_make_joint_action(0.03),
            waypoint_action=_MISSING,
            is_gripper_close=True,
        )

    with h5py.File(h5_path, "r") as h5:
        assert select_keyframe_jointangle.list_episode_indices(h5) == [1, 2, 10]

        episode = h5["episode_2"]
        assert select_keyframe_jointangle.iter_sorted_timestep_keys(episode) == [
            "timestep_0",
            "timestep_1",
            "timestep_2",
            "timestep_3",
            "timestep_3_dup1",
            "timestep_4",
            "timestep_5",
            "timestep_6",
            "timestep_7",
        ]

        scan = select_keyframe_jointangle.scan_episode_keyframes(
            episode,
            motion_low=0.002,
            motion_high=0.005,
        )

    assert scan.valid_waypoint_count == 6
    assert scan.valid_joint_count == 8
    assert scan.predicted_keyframes == [1, 4, 6]
    assert scan.native_keyframes == [0, 3, 6, 7]
    assert scan.frames[0].is_video_demo is True
    assert scan.frames[2].waypoint_action is None
    assert scan.frames[7].joint_action is None


def test_render_comparison_frame_highlights_predicted_and_native_independently() -> None:
    base_rgb = _make_rgb(120)
    record = select_keyframe_jointangle.FrameRecord(
        frame_index=6,
        timestep_key="timestep_6",
        is_video_demo=False,
        waypoint_action=None,
        joint_action=_make_joint_action(0.03),
        is_gripper_close=True,
    )

    predicted_only = select_keyframe_jointangle.render_comparison_frame(
        base_rgb,
        task_name="Dummy",
        episode=0,
        record=record,
        predicted_keyframes={6: 1},
        native_keyframes={},
    )
    native_only = select_keyframe_jointangle.render_comparison_frame(
        base_rgb,
        task_name="Dummy",
        episode=0,
        record=record,
        predicted_keyframes={},
        native_keyframes={6: 1},
    )

    predicted_color = np.array(select_keyframe_jointangle.PREDICTED_COLOR_BGR, dtype=np.uint8)
    native_color = np.array(select_keyframe_jointangle.NATIVE_COLOR_BGR, dtype=np.uint8)
    base_color = np.array([120, 120, 120], dtype=np.uint8)

    assert predicted_only.shape == (64, 32, 3)
    assert np.array_equal(predicted_only[0, 0], predicted_color)
    assert np.array_equal(predicted_only[32, 0], base_color)

    assert native_only.shape == (64, 32, 3)
    assert np.array_equal(native_only[0, 0], base_color)
    assert np.array_equal(native_only[32, 0], native_color)


def test_export_episode_outputs_writes_png_json_and_compare_video(tmp_path: Path) -> None:
    h5_path = tmp_path / "record_dataset_Dummy.h5"
    waypoint_a = np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float64)
    waypoint_b = np.array([10, 20, 30, 40, 50, 60, 1], dtype=np.float64)

    with h5py.File(h5_path, "w") as h5:
        episode = h5.create_group("episode_0")
        _make_timestep(
            episode,
            "timestep_0",
            fill_value=0,
            joint_action=_make_joint_action(0.0),
            waypoint_action=waypoint_a,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_1",
            fill_value=1,
            joint_action=_make_joint_action(0.0),
            waypoint_action=waypoint_a,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_2",
            fill_value=2,
            joint_action=_make_joint_action(0.01),
            waypoint_action=waypoint_b,
            is_gripper_close=False,
        )
        _make_timestep(
            episode,
            "timestep_3",
            fill_value=3,
            joint_action=_make_joint_action(0.01),
            waypoint_action=waypoint_b,
            is_gripper_close=True,
        )
        _make_timestep(
            episode,
            "timestep_4",
            fill_value=4,
            joint_action=_make_joint_action(0.01),
            waypoint_action=waypoint_b,
            is_gripper_close=True,
        )

    output_dir = tmp_path / "out"
    summary = select_keyframe_jointangle.export_episode_outputs(
        h5_path,
        task_name="Dummy",
        episode=0,
        output_dir=output_dir,
        fps=5.0,
        motion_low=0.002,
        motion_high=0.005,
        match_tolerance=2,
    )

    assert summary.timeline_png_path.name == "Dummy_ep0_keyframe_timeline.png"
    assert summary.timeline_json_path.name == "Dummy_ep0_keyframe_timeline.json"
    assert summary.compare_video_path.name == "Dummy_ep0_front_rgb_keyframe_compare.mp4"
    assert summary.frame_count == 5
    assert summary.predicted_keyframe_count == 2
    assert summary.native_keyframe_count == 2

    for path in [
        summary.timeline_png_path,
        summary.timeline_json_path,
        summary.compare_video_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0

    assert not (output_dir / "Dummy_ep0_front_rgb_keyframes.mp4").exists()

    payload = json.loads(summary.timeline_json_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "Dummy"
    assert payload["episode"] == 0
    assert payload["frame_count"] == 5
    assert payload["predicted"]["count"] == 2
    assert payload["predicted"]["frame_indices"] == [1, 3]
    assert payload["predicted"]["timestep_keys"] == ["timestep_1", "timestep_3"]
    assert payload["native"]["count"] == 2
    assert payload["native"]["frame_indices"] == [0, 2]
    assert payload["native"]["timestep_keys"] == ["timestep_0", "timestep_2"]
    assert payload["comparison"]["tp"] == 2
    assert payload["comparison"]["fp"] == 0
    assert payload["comparison"]["fn"] == 0
    assert payload["comparison"]["matched_pairs"] == [
        {
            "predicted_frame_index": 1,
            "predicted_timestep_key": "timestep_1",
            "native_frame_index": 0,
            "native_timestep_key": "timestep_0",
            "distance": 1,
        },
        {
            "predicted_frame_index": 3,
            "predicted_timestep_key": "timestep_3",
            "native_frame_index": 2,
            "native_timestep_key": "timestep_2",
            "distance": 1,
        },
    ]
