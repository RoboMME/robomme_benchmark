import importlib.util
import json
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


matcher_mod = _load_module(
    "oracle_action_matcher_under_test",
    "src/robomme/robomme_env/utils/oracle_action_matcher.py",
)
resolver_mod = _load_module(
    "episode_dataset_resolver_under_test",
    "src/robomme/env_record_wrapper/episode_dataset_resolver.py",
)


def test_find_exact_label_option_index_matches_label_only():
    options = [
        {"label": "a", "action": "pick up the cube"},
        {"label": "b", "action": "put it down"},
    ]

    assert matcher_mod.find_exact_label_option_index("a", options) == 0
    assert matcher_mod.find_exact_label_option_index("b", options) == 1
    assert matcher_mod.find_exact_label_option_index("pick up the cube", options) == -1
    assert matcher_mod.find_exact_label_option_index(1, options) == -1


def test_map_action_text_to_option_label_strict_exact():
    options = [
        {"label": "a", "action": "pick up the cube"},
        {"label": "b", "action": "put it down"},
    ]

    assert (
        matcher_mod.map_action_text_to_option_label("pick up the cube", options) == "a"
    )
    assert matcher_mod.map_action_text_to_option_label("unknown action", options) is None
    assert matcher_mod.map_action_text_to_option_label(None, options) is None


def test_episode_dataset_resolver_extracts_choice_command_and_ignores_empty_choice(tmp_path):
    h5_path = tmp_path / "choice_oracle_commands.h5"

    with h5py.File(h5_path, "w") as h5:
        episode_group = h5.create_group("episode_0")

        ts0 = episode_group.create_group("timestep_0")
        ts0_action = ts0.create_group("action")
        ts0_action.create_dataset(
            "choice_action",
            data=json.dumps(
                {
                    "choice": "B",
                    "point": [34, 12],
                }
            ),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        ts0_info = ts0.create_group("info")
        ts0_info.create_dataset("is_video_demo", data=False)
        ts0_info.create_dataset("is_subgoal_boundary", data=True)

        ts1 = episode_group.create_group("timestep_1")
        ts1_action = ts1.create_group("action")
        ts1_action.create_dataset(
            "choice_action",
            data=json.dumps(
                {
                    "choice": "",
                    "point": [30, 20],
                }
            ),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        ts1_info = ts1.create_group("info")
        ts1_info.create_dataset("is_video_demo", data=False)
        ts1_info.create_dataset("is_subgoal_boundary", data=True)

    resolver = resolver_mod.EpisodeDatasetResolver(
        env_id="DummyEnv",
        episode=0,
        dataset_directory=h5_path,
    )
    try:
        command0 = resolver.get_step("multi_choice", 0)
        assert command0 == {"choice": "B", "point": [34.0, 12.0]}
        assert "position_3d" not in command0

        command1 = resolver.get_step("multi_choice", 1)
        assert command1 is None
    finally:
        resolver.close()
