from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests._shared.repo_paths import find_repo_root, ensure_src_on_path

_PROJECT_ROOT = find_repo_root(__file__)
ensure_src_on_path(__file__)

_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "dev"
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
if _SCRIPT_DIR_STR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_STR)

import snapshot as snapshot_utils  # noqa: E402


class _FakeActor:
    def __init__(self, name: str, position_xyz):
        self.name = name
        self.pose = SimpleNamespace(
            p=np.asarray(position_xyz, dtype=np.float32).reshape(1, 3)
        )


def test_collect_button_unmask_swap_snapshot_shape() -> None:
    bin_0 = _FakeActor("bin_0", [0.10, -0.10, 0.052])
    bin_1 = _FakeActor("bin_1", [0.00, 0.15, 0.052])
    bin_2 = _FakeActor("bin_2", [0.13, 0.16, 0.052])
    bin_3 = _FakeActor("bin_3", [0.09, -0.03, 0.052])

    cube_blue = _FakeActor("target_cube_blue", [0.10, -0.10, 0.0167])
    cube_red = _FakeActor("target_cube_red", [0.13, 0.16, 0.0167])
    cube_green = _FakeActor("target_cube_green", [0.00, 0.15, 0.0167])

    base_env = SimpleNamespace(
        spawned_bins=[bin_0, bin_1, bin_2, bin_3],
        cube_bin_pairs=[
            (cube_blue, bin_0),
            (cube_red, bin_2),
            (cube_green, bin_1),
        ],
        color_names=["blue", "red", "green"],
        bin_to_color={0: "blue", 2: "red", 1: "green"},
    )

    payload = snapshot_utils._collect_snapshot(
        base_env=base_env,
        env_id="ButtonUnmaskSwap",
        episode=1,
        seed=0,
        difficulty="hard",
        capture_elapsed_steps=33,
    )

    assert payload["env_id"] == "ButtonUnmaskSwap"
    assert payload["inspect_this_timestep"] == 33
    assert payload["capture_elapsed_steps"] == 33

    cubes = payload["cubes"]
    bins = payload["bins"]

    assert len(cubes) == len(base_env.cube_bin_pairs)
    assert len(bins) == len(base_env.spawned_bins)

    for cube_item in cubes:
        assert "color" in cube_item
        assert isinstance(cube_item["position_xyz"], list)
        assert len(cube_item["position_xyz"]) == 3

    for bin_item in bins:
        assert "color" not in bin_item
        assert isinstance(bin_item["position_xyz"], list)
        assert len(bin_item["position_xyz"]) == 3

    empty_bin = next(item for item in bins if item["index"] == 3)
    assert empty_bin["has_cube_under_bin"] is False


def test_snapshot_json_path_location() -> None:
    output_root = Path("/tmp/robomme_snapshot_test")
    json_path = snapshot_utils._snapshot_json_path(
        output_root=output_root,
        env_id="ButtonUnmaskSwap",
        episode=1,
        seed=7,
    )
    assert json_path == output_root / "snapshots" / "ButtonUnmaskSwap_ep1_seed7_after_drop.json"


def test_button_unmask_swap_inspect_this_timestep_value() -> None:
    assert snapshot_utils.SNAPSHOT_ENVS["ButtonUnmaskSwap"] == 33
