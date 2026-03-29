import importlib.util
import sys

import matplotlib.pyplot as plt
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


aggregate_mod = _load_module(
    "aggregate_visible_object_xy_under_test",
    "scripts/dev2/aggregate-visible-object-xy.py",
)


def _point(*, env_id: str, name: str, semantic: str) -> object:
    return aggregate_mod.VisibleObjectPoint(
        env_id=env_id,
        episode=0,
        seed=0,
        name=name,
        world_x=0.1,
        world_y=0.2,
        world_z=0.3,
        semantic=semantic,
        cube_color=aggregate_mod._cube_color(name),
        button_kind=aggregate_mod._button_kind(name),
        bin_index=aggregate_mod._bin_index(name),
    )


def test_semantic_category_recognizes_binfill_board_with_hole_as_target() -> None:
    assert aggregate_mod._semantic_category("BinFill", "board_with_hole") == "target"


def test_semantic_category_recognizes_bin_for_selected_unmask_envs() -> None:
    for env_id in aggregate_mod.BIN_PANEL_ENVS:
        assert aggregate_mod._semantic_category(env_id, "bin_0") == "bin"


def test_semantic_category_keeps_bin_as_other_for_non_selected_envs() -> None:
    assert aggregate_mod._semantic_category("MoveCube", "bin_0") == "other"


def test_semantic_category_keeps_existing_cube_and_button_rules() -> None:
    assert aggregate_mod._semantic_category("BinFill", "cube_red_0") == "cube"
    assert aggregate_mod._semantic_category("BinFill", "button_cap") == "button"
    assert aggregate_mod._semantic_category("MoveCube", "board_with_hole") == "other"


def test_bin_index_extracts_numeric_suffix() -> None:
    assert aggregate_mod._bin_index("bin_0") == 0
    assert aggregate_mod._bin_index("bin_12") == 12
    assert aggregate_mod._bin_index("bin_x") is None
    assert aggregate_mod._bin_index("button_bin_0") is None


def test_target_panel_points_for_binfill_only_keeps_board_with_hole() -> None:
    points = [
        _point(env_id="BinFill", name="board_with_hole", semantic="target"),
        _point(env_id="BinFill", name="target_marker", semantic="target"),
        _point(env_id="BinFill", name="cube_red_0", semantic="cube"),
    ]

    panel_points = aggregate_mod._target_panel_points("BinFill", points)

    assert [point.name for point in panel_points] == ["board_with_hole"]


def test_target_panel_points_for_other_env_use_generic_target_semantic() -> None:
    points = [
        _point(env_id="MoveCube", name="target_a", semantic="target"),
        _point(env_id="MoveCube", name="board_with_hole", semantic="other"),
        _point(env_id="MoveCube", name="cube_red_0", semantic="cube"),
    ]

    panel_points = aggregate_mod._target_panel_points("MoveCube", points)

    assert [point.name for point in panel_points] == ["target_a"]


def test_plot_target_objects_uses_binfill_board_with_hole_title() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_target_objects(
            ax,
            "BinFill",
            [
                _point(env_id="BinFill", name="board_with_hole", semantic="target"),
                _point(env_id="BinFill", name="target_marker", semantic="target"),
            ],
        )
        assert ax.get_title().startswith("board_with_hole (Rotated XY)")
    finally:
        plt.close(fig)


def test_plot_bin_objects_uses_bin_title_and_index_legend() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_bin_objects(
            ax,
            [
                _point(env_id="VideoUnmask", name="bin_0", semantic="bin"),
                _point(env_id="VideoUnmask", name="bin_1", semantic="bin"),
            ],
        )
        assert ax.get_title().startswith("Bin (Rotated XY)")
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["bin_0", "bin_1"]
    finally:
        plt.close(fig)
