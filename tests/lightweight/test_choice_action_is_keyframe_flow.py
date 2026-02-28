# -*- coding: utf-8 -*-
"""
轻量测试：choice_action 记录格式 + EpisodeDatasetResolver 按 info/is_keyframe 读取。

运行方式（使用 uv）：
    uv run python tests/lightweight/test_choice_action_is_keyframe_flow.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import h5py
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = [pytest.mark.lightweight, pytest.mark.gpu]


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolver_mod = _load_module(
    "episode_dataset_resolver_under_test",
    "src/robomme/env_record_wrapper/episode_dataset_resolver.py",
)


def _make_timestep(
    episode_group: h5py.Group,
    timestep_idx: int,
    *,
    choice_action: dict | None = None,
    is_video_demo: bool = False,
    is_keyframe: bool = False,
) -> None:
    ts = episode_group.create_group(f"timestep_{timestep_idx}")
    action = ts.create_group("action")
    payload = "{}" if choice_action is None else json.dumps(choice_action)
    action.create_dataset(
        "choice_action",
        data=payload,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)
    info.create_dataset("is_keyframe", data=is_keyframe)


def _build_h5(h5_path: Path) -> None:
    with h5py.File(h5_path, "w") as h5:
        ep = h5.create_group("episode_0")

        # 非 keyframe: 有效 label 也必须忽略
        _make_timestep(
            ep,
            0,
            choice_action={
                "label": "a",
                "position": [10, 20],
                "position_3d": [0.1, 0.2, 0.3],
            },
            is_keyframe=False,
        )
        # 有效 keyframe: 应被读取
        _make_timestep(
            ep,
            1,
            choice_action={
                "label": "b",
                "position": [12, 34],
                "position_3d": [1.2, 3.4, 5.6],
            },
            is_keyframe=True,
        )
        # keyframe 但空标签: 跳过
        _make_timestep(
            ep,
            2,
            choice_action={
                "label": "",
                "position": [20, 30],
                "position_3d": [2.0, 3.0, 4.0],
            },
            is_keyframe=True,
        )
        # video demo keyframe: 跳过
        _make_timestep(
            ep,
            3,
            choice_action={
                "label": "c",
                "position": [70, 80],
                "position_3d": [7.0, 8.0, 9.0],
            },
            is_video_demo=True,
            is_keyframe=True,
        )
        # 第二个有效 keyframe
        _make_timestep(
            ep,
            4,
            choice_action={
                "label": "d",
                "position": [90, 11],
                "position_3d": [9.0, 1.1, 2.2],
            },
            is_keyframe=True,
        )


def _assert_record_schema_contract(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5:
        ts1 = h5["episode_0"]["timestep_1"]
        raw = ts1["action"]["choice_action"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        assert "serial_number" not in payload, "choice_action should not store serial_number"
        assert payload["label"] == "b"
        assert payload["position"] == [12, 34]
        assert payload["position_3d"] == [1.2, 3.4, 5.6]
        assert bool(ts1["info"]["is_keyframe"][()]) is True


def _assert_resolver_reads_by_is_keyframe(h5_path: Path) -> None:
    resolver = resolver_mod.EpisodeDatasetResolver(
        env_id="Dummy",
        episode=0,
        dataset_directory=str(h5_path),
    )
    try:
        assert resolver.get_step("multi_choice", -1) is None

        command0 = resolver.get_step("multi_choice", 0)
        assert command0 == {"label": "b", "position": [12.0, 34.0]}
        assert "position_3d" not in command0
        assert "serial_number" not in command0

        command1 = resolver.get_step("multi_choice", 1)
        assert command1 == {"label": "d", "position": [90.0, 11.0]}
        assert "position_3d" not in command1
        assert "serial_number" not in command1

        assert resolver.get_step("multi_choice", 2) is None
    finally:
        resolver.close()


def test_choice_action_is_keyframe_flow_pytest(tmp_path: Path) -> None:
    h5_path = tmp_path / "choice_action_flow.h5"
    _build_h5(h5_path)
    _assert_record_schema_contract(h5_path)
    _assert_resolver_reads_by_is_keyframe(h5_path)


def main() -> None:
    print("\n[TEST] choice_action is_keyframe flow")
    with tempfile.TemporaryDirectory(prefix="choice_action_is_keyframe_") as tmp:
        h5_path = Path(tmp) / "choice_action_flow.h5"
        _build_h5(h5_path)
        _assert_record_schema_contract(h5_path)
        print("  schema ✓ choice_action 无 serial_number, is_keyframe 可读")

        _assert_resolver_reads_by_is_keyframe(h5_path)
        print("  resolver ✓ 仅按 is_keyframe 读取 + 跳过空标签/video_demo")

    print("\nPASS: choice_action is_keyframe flow tests passed")


if __name__ == "__main__":
    main()
