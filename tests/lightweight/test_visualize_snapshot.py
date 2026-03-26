from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root, ensure_src_on_path

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)

_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "dev"
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
if _SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_STR)

import visualize as visualize_utils  # noqa: E402


def _write_snapshot_json(
    path: Path,
    *,
    env_id: str,
    episode: int,
    seed: int,
    include_buttons: bool,
) -> None:
    payload = {
        "env_id": env_id,
        "episode": episode,
        "seed": seed,
        "difficulty": "easy",
        "inspect_this_timestep": 33,
        "capture_elapsed_steps": 33,
        "capture_phase": "after_drop",
        "collision": False,
        "bins": [
            {
                "index": 0,
                "name": "bin_0",
                "position_xyz": [0.1, -0.1, 0.052],
                "has_cube_under_bin": True,
            },
            {
                "index": 1,
                "name": "bin_1",
                "position_xyz": [0.0, 0.15, 0.052],
                "has_cube_under_bin": False,
            },
        ],
        "cubes": [
            {
                "name": "target_cube_red",
                "color": "red",
                "position_xyz": [0.1, -0.1, 0.0167],
                "paired_bin_index": 0,
                "paired_bin_name": "bin_0",
            }
        ],
    }
    if include_buttons:
        payload["buttons"] = [
            {
                "name": "button_left",
                "position_xyz": [-0.2, 0.0, 0.052],
            }
        ]

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_snapshot_accepts_optional_buttons(tmp_path) -> None:
    button_snapshot_path = tmp_path / "ButtonUnmask_ep0_seed1_after_drop.json"
    video_snapshot_path = tmp_path / "VideoUnmask_ep0_seed2_after_drop.json"
    _write_snapshot_json(
        button_snapshot_path,
        env_id="ButtonUnmask",
        episode=0,
        seed=1,
        include_buttons=True,
    )
    _write_snapshot_json(
        video_snapshot_path,
        env_id="VideoUnmask",
        episode=0,
        seed=2,
        include_buttons=False,
    )

    button_scene = visualize_utils._load_snapshot(button_snapshot_path)
    video_scene = visualize_utils._load_snapshot(video_snapshot_path)

    assert len(button_scene.buttons) == 1
    assert button_scene.buttons[0].name == "button_left"
    assert video_scene.buttons == ()


def test_save_figure_succeeds_for_mixed_button_snapshots(tmp_path) -> None:
    input_dir = tmp_path / "snapshots"
    input_dir.mkdir()

    _write_snapshot_json(
        input_dir / "ButtonUnmask_ep0_seed1_after_drop.json",
        env_id="ButtonUnmask",
        episode=0,
        seed=1,
        include_buttons=True,
    )
    _write_snapshot_json(
        input_dir / "VideoUnmask_ep0_seed2_after_drop.json",
        env_id="VideoUnmask",
        episode=0,
        seed=2,
        include_buttons=False,
    )

    scenes = visualize_utils._load_snapshots(sorted(input_dir.glob("*.json")))
    grouped_scenes = visualize_utils._group_scenes_by_env(scenes)

    saved_paths: list[Path] = []
    for env_id, env_scenes in grouped_scenes.items():
        output_path = tmp_path / f"{env_id}.png"
        visualize_utils._save_figure(
            env_scenes,
            output_path=output_path,
            dpi=100,
            env_id=env_id,
        )
        saved_paths.append(output_path)

    assert len(saved_paths) == 2
    assert all(path.exists() for path in saved_paths)
