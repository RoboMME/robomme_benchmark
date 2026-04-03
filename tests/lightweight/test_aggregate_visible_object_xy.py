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


def _point(
    *,
    env_id: str,
    name: str,
    semantic: str,
    episode: int = 0,
    seed: int = 0,
    world_x: float = 0.1,
    world_y: float = 0.2,
) -> object:
    return aggregate_mod.VisibleObjectPoint(
        env_id=env_id,
        episode=episode,
        seed=seed,
        name=name,
        world_x=world_x,
        world_y=world_y,
        world_z=0.3,
        semantic=semantic,
        cube_color=aggregate_mod._cube_color(name),
        button_kind=aggregate_mod._button_kind(name),
        bin_index=aggregate_mod._bin_index(name),
        peg_part=aggregate_mod._peg_part(name),
    )


def test_semantic_category_recognizes_binfill_board_with_hole_as_target() -> None:
    assert aggregate_mod._semantic_category("BinFill", "board_with_hole") == "target"


def test_semantic_category_recognizes_bin_for_selected_unmask_envs() -> None:
    for env_id in aggregate_mod.BIN_PANEL_ENVS:
        assert aggregate_mod._semantic_category(env_id, "bin_0") == "bin"


def test_semantic_category_recognizes_insertpeg_and_movecube_groups() -> None:
    assert aggregate_mod._semantic_category("InsertPeg", "peg_head") == "peg"
    assert aggregate_mod._semantic_category("InsertPeg", "box_with_hole") == "box_with_hole"
    assert aggregate_mod._semantic_category("MoveCube", "goal_site") == "goal_site"
    assert aggregate_mod._semantic_category("MoveCube", "fixed_cube") == "cube"


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


def test_peg_part_extracts_named_peg_parts() -> None:
    assert aggregate_mod._peg_part("peg_head") == "peg_head"
    assert aggregate_mod._peg_part("peg_tail") == "peg_tail"
    assert aggregate_mod._peg_part("goal_site") is None


def test_cube_label_uses_obstacle_for_routestick_unknown_only() -> None:
    assert aggregate_mod._cube_label("RouteStick", "unknown") == "obstacle"
    assert aggregate_mod._cube_label("MoveCube", "unknown") == "cube_unknown"
    assert aggregate_mod._cube_label("RouteStick", "red") == "cube_red"


def test_panel_specs_match_insertpeg_and_movecube_layouts() -> None:
    assert aggregate_mod._panel_specs_for_env("InsertPeg") == (
        "all",
        "peg",
        "box_with_hole",
    )
    assert aggregate_mod._panel_specs_for_env("MoveCube") == (
        "all",
        "peg",
        "goal_site",
        "cube",
    )


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


def test_plot_cube_objects_uses_obstacle_label_for_routestick_unknown() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_cube_objects(
            ax,
            "RouteStick",
            [_point(env_id="RouteStick", name="target_cube_1", semantic="cube")],
        )
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["obstacle"]
    finally:
        plt.close(fig)


def test_plot_peg_objects_uses_title_and_legend() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_peg_objects(
            ax,
            [
                _point(env_id="InsertPeg", name="peg_head", semantic="peg"),
                _point(env_id="InsertPeg", name="peg_tail", semantic="peg"),
            ],
        )
        assert ax.get_title().startswith("Peg Head/Tail (Rotated XY)")
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["peg_head", "peg_tail"]
    finally:
        plt.close(fig)


def test_nearest_episode_peg_pairs_match_same_episode_by_nearest_distance() -> None:
    pairs = aggregate_mod._nearest_episode_peg_pairs(
        [
            _point(
                env_id="InsertPeg",
                name="peg_head",
                semantic="peg",
                episode=0,
                world_x=0.00,
                world_y=0.00,
            ),
            _point(
                env_id="InsertPeg",
                name="peg_head",
                semantic="peg",
                episode=0,
                world_x=1.00,
                world_y=1.00,
            ),
            _point(
                env_id="InsertPeg",
                name="peg_tail",
                semantic="peg",
                episode=0,
                world_x=0.05,
                world_y=0.00,
            ),
            _point(
                env_id="InsertPeg",
                name="peg_tail",
                semantic="peg",
                episode=0,
                world_x=0.95,
                world_y=1.00,
            ),
            _point(
                env_id="InsertPeg",
                name="peg_head",
                semantic="peg",
                episode=1,
                world_x=-0.10,
                world_y=0.20,
            ),
            _point(
                env_id="InsertPeg",
                name="peg_tail",
                semantic="peg",
                episode=1,
                world_x=-0.12,
                world_y=0.22,
            ),
        ]
    )

    assert len(pairs) == 3
    pair_positions = {
        ((head.episode, head.seed), round(head.world_x, 2), round(tail.world_x, 2))
        for head, tail in pairs
    }
    assert pair_positions == {
        ((0, 0), 0.00, 0.05),
        ((0, 0), 1.00, 0.95),
        ((1, 0), -0.10, -0.12),
    }


def test_plot_peg_objects_adds_one_line_per_episode_pair() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_peg_objects(
            ax,
            [
                _point(
                    env_id="InsertPeg",
                    name="peg_head",
                    semantic="peg",
                    episode=0,
                    world_x=0.00,
                    world_y=0.00,
                ),
                _point(
                    env_id="InsertPeg",
                    name="peg_tail",
                    semantic="peg",
                    episode=0,
                    world_x=0.01,
                    world_y=0.00,
                ),
                _point(
                    env_id="InsertPeg",
                    name="peg_head",
                    semantic="peg",
                    episode=1,
                    world_x=0.10,
                    world_y=0.10,
                ),
                _point(
                    env_id="InsertPeg",
                    name="peg_tail",
                    semantic="peg",
                    episode=1,
                    world_x=0.11,
                    world_y=0.10,
                ),
            ],
        )
        assert len(ax.lines) == 2
    finally:
        plt.close(fig)


def test_plot_goal_site_objects_uses_goal_site_title() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_goal_site_objects(
            ax,
            [_point(env_id="MoveCube", name="goal_site", semantic="goal_site")],
        )
        assert ax.get_title().startswith("goal_site (Rotated XY)")
    finally:
        plt.close(fig)


def test_plot_box_with_hole_objects_uses_box_title() -> None:
    fig, ax = plt.subplots()
    try:
        aggregate_mod._plot_box_with_hole_objects(
            ax,
            [_point(env_id="InsertPeg", name="box_with_hole", semantic="box_with_hole")],
        )
        assert ax.get_title().startswith("box_with_hole (Rotated XY)")
    finally:
        plt.close(fig)
