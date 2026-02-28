import importlib.util
from pathlib import Path

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


mapping_mod = _load_module(
    "choice_action_mapping_under_test",
    "src/robomme/robomme_env/utils/choice_action_mapping.py",
)


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


def _camera_params():
    intrinsic = [
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 40.0],
        [0.0, 0.0, 1.0],
    ]
    extrinsic = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    image_shape = (100, 120, 3)
    return intrinsic, extrinsic, image_shape


def test_project_world_to_pixel_basic_projection():
    intrinsic, extrinsic, image_shape = _camera_params()
    projected = mapping_mod.project_world_to_pixel(
        world_xyz=[0.2, 0.1, 1.0],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )
    assert projected == [70, 50]


def test_project_world_to_pixel_returns_none_on_invalid_projection():
    intrinsic, extrinsic, image_shape = _camera_params()
    # Out of image bound.
    out_of_bounds = mapping_mod.project_world_to_pixel(
        world_xyz=[2.0, 0.0, 1.0],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )
    assert out_of_bounds is None

    # Behind camera.
    behind_camera = mapping_mod.project_world_to_pixel(
        world_xyz=[0.0, 0.0, -1.0],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )
    assert behind_camera is None


def test_select_target_with_pixel_picks_nearest_projected_candidate():
    intrinsic, extrinsic, image_shape = _camera_params()
    actor_a = _Actor("A", [0.0, 0.0, 1.0])   # pixel [50, 40]
    actor_b = _Actor("B", [0.2, 0.1, 1.0])   # pixel [70, 50]
    actor_c = _Actor("C", [-0.2, 0.0, 1.0])  # pixel [30, 40]

    result = mapping_mod.select_target_with_pixel(
        available=[actor_a, actor_b, actor_c],
        pixel_like=[68, 49],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )

    assert result is not None
    assert result["name"] == "B"
    assert result["projected_pixel"] == [70, 50]
    assert result["selection_mode"] == "nearest_pixel_projection"


def test_select_target_with_pixel_returns_none_for_invalid_inputs():
    intrinsic, extrinsic, image_shape = _camera_params()
    actor_invalid = _Actor("invalid", None)

    invalid_pixel = mapping_mod.select_target_with_pixel(
        available=[_Actor("valid", [0.0, 0.0, 1.0])],
        pixel_like=[10],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )
    assert invalid_pixel is None

    invalid_actor = mapping_mod.select_target_with_pixel(
        available=[actor_invalid],
        pixel_like=[10, 10],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )
    assert invalid_actor is None


def test_select_target_with_pixel_tie_breaks_by_first_flattened_candidate():
    intrinsic, extrinsic, image_shape = _camera_params()
    actor_first = _Actor("first", [0.1, 0.0, 1.0])   # pixel [60, 40]
    actor_second = _Actor("second", [-0.1, 0.0, 1.0])  # pixel [40, 40]

    result = mapping_mod.select_target_with_pixel(
        available={"left": [actor_first], "right": [actor_second]},
        pixel_like=[50, 40],
        intrinsic_cv=intrinsic,
        extrinsic_cv=extrinsic,
        image_shape=image_shape,
    )

    assert result is not None
    assert result["name"] == "first"
