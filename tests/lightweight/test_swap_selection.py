import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tests._shared.repo_paths import find_repo_root


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


swap_mod = _load_module(
    "swap_selection_under_test",
    "src/robomme/robomme_env/utils/swap_selection.py",
)

pytestmark = [pytest.mark.lightweight]


class _Pose:
    def __init__(self, p):
        self.p = p


class _Actor:
    def __init__(self, name, pose_p=None, use_get_pose=False):
        self.name = name
        self._pose = _Pose(pose_p) if pose_p is not None else None
        if not use_get_pose and self._pose is not None:
            self.pose = self._pose

    def get_pose(self):
        if self._pose is None:
            raise RuntimeError("missing pose")
        return self._pose


def _actors_from_xy(points):
    return [_Actor(f"actor_{idx}", [x, y, 0.0]) for idx, (x, y) in enumerate(points)]


def _find_seed_for_pair(candidates, target_pair, limit=2048):
    for seed in range(limit):
        generator = torch.Generator().manual_seed(seed)
        result = swap_mod.select_dynamic_swap_pair(candidates, generator=generator)
        if result is not None and result["pair_key"] == target_pair:
            return seed
    raise AssertionError(f"Could not find seed for pair {target_pair}")


def test_extract_candidate_positions_skips_invalid_candidates():
    positions = swap_mod.extract_candidate_positions(
        [
            _Actor("valid_pose", [1.0, 2.0, 3.0]),
            _Actor("valid_get_pose", [4.0, 5.0, 6.0], use_get_pose=True),
            _Actor("missing_pose"),
            _Actor("short_pose", [7.0]),
        ]
    )

    assert np.allclose(positions[0], [1.0, 2.0, 3.0])
    assert np.allclose(positions[1], [4.0, 5.0, 6.0])
    assert positions[2] is None
    assert positions[3] is None


def test_select_dynamic_swap_pair_is_reproducible_and_sorted():
    actors = _actors_from_xy([(0.0, 0.0), (0.2, 0.0), (1.0, 0.0), (3.0, 0.0)])

    result_a = swap_mod.select_dynamic_swap_pair(
        actors, generator=torch.Generator().manual_seed(7)
    )
    result_b = swap_mod.select_dynamic_swap_pair(
        actors, generator=torch.Generator().manual_seed(7)
    )

    assert result_a == result_b
    assert result_a is not None
    assert result_a["pair_key"] == tuple(sorted((result_a["idx1"], result_a["idx2"])))

    idx1, idx2 = result_a["idx1"], result_a["idx2"]
    expected_distance = float(
        np.linalg.norm(np.array(actors[idx1].pose.p[:2]) - np.array(actors[idx2].pose.p[:2]))
    )
    assert abs(result_a["distance"] - expected_distance) < 1e-9


def test_select_dynamic_swap_pair_falls_back_to_nearest_legal_pair():
    actors = _actors_from_xy([(0.0, 0.0), (0.1, 0.0), (5.0, 0.0), (5.2, 0.0)])
    repeated_pair = (0, 1)
    seed = _find_seed_for_pair(actors, repeated_pair)

    result = swap_mod.select_dynamic_swap_pair(
        actors,
        generator=torch.Generator().manual_seed(seed),
        previous_pair=repeated_pair,
    )

    assert result is not None
    assert result["pair_key"] == (2, 3)
    assert abs(result["distance"] - 0.2) < 1e-6


def test_select_dynamic_swap_pair_allows_repeat_only_when_no_new_pair_exists():
    actors = _actors_from_xy([(0.0, 0.0), (0.2, 0.0)])
    repeated_pair = (0, 1)
    seed = _find_seed_for_pair(actors, repeated_pair)

    result = swap_mod.select_dynamic_swap_pair(
        actors,
        generator=torch.Generator().manual_seed(seed),
        previous_pair=repeated_pair,
    )

    assert result is not None
    assert result["pair_key"] == repeated_pair


def test_select_dynamic_swap_pair_skips_invalid_candidates():
    actors = [
        _Actor("invalid"),
        _Actor("valid_a", [0.0, 0.0, 0.0]),
        _Actor("valid_b", [0.3, 0.0, 0.0], use_get_pose=True),
    ]

    result = swap_mod.select_dynamic_swap_pair(
        actors, generator=torch.Generator().manual_seed(3)
    )

    assert result is not None
    assert result["pair_key"] == (1, 2)
