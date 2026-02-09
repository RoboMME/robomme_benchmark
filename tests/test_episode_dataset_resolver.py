from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "historybench/env_record_wrapper/episode_dataset_resolver.py"
MODULE_SPEC = spec_from_file_location("episode_dataset_resolver", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
MODULE = module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

EpisodeDatasetResolver = MODULE.EpisodeDatasetResolver
list_episode_indices = MODULE.list_episode_indices


def _add_timestep(
    episode_group: h5py.Group,
    record_step: int,
    *,
    demonstration: bool,
    action=None,
    keypoint_p=None,
    keypoint_q=None,
    robot_endeffector_p=None,
    robot_endeffector_q=None,
    grounded_subgoal=None,
):
    g = episode_group.create_group(f"record_timestep_{record_step}")
    g.create_dataset("demonstration", data=np.array(demonstration))
    if action is not None:
        g.create_dataset("action", data=action)
    if keypoint_p is not None:
        g.create_dataset("keypoint_p", data=np.asarray(keypoint_p, dtype=np.float64))
    if keypoint_q is not None:
        g.create_dataset("keypoint_q", data=np.asarray(keypoint_q, dtype=np.float64))
    if robot_endeffector_p is not None:
        g.create_dataset("robot_endeffector_p", data=np.asarray(robot_endeffector_p, dtype=np.float64))
    if robot_endeffector_q is not None:
        g.create_dataset("robot_endeffector_q", data=np.asarray(robot_endeffector_q, dtype=np.float64))
    if grounded_subgoal is not None:
        g.create_dataset("grounded_subgoal", data=grounded_subgoal)


def _build_test_h5(file_path: Path, env_id: str = "TestEnv"):
    with h5py.File(file_path, "w") as h5:
        env_group = h5.create_group(f"env_{env_id}")
        episode0 = env_group.create_group("episode_0")
        env_group.create_group("episode_2")

        _add_timestep(
            episode0,
            0,
            demonstration=True,
            action=np.array([99, 99, 99, 99, 99, 99, 99], dtype=np.float64),
            grounded_subgoal="demo",
        )
        _add_timestep(
            episode0,
            1,
            demonstration=False,
            action=np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float64),
            keypoint_p=[0.1, 0.2, 0.3],
            keypoint_q=[0.0, 0.0, 0.0, 1.0],
            robot_endeffector_p=[1, 2, 3],
            robot_endeffector_q=[0.0, 0.0, 0.0, 1.0],
            grounded_subgoal="alpha",
        )
        _add_timestep(
            episode0,
            2,
            demonstration=False,
            action=np.array([8, 9], dtype=np.float64),
            robot_endeffector_p=[4, 5, 6],
            grounded_subgoal="alpha",
        )
        _add_timestep(
            episode0,
            3,
            demonstration=False,
            action="None",
            keypoint_p=[0.4, 0.5, 0.6],
            keypoint_q=[0.0, 0.0, 1.0, 0.0],
            robot_endeffector_p=[7, 8, 9],
            robot_endeffector_q=[0.0, 0.0, 1.0, 0.0],
            grounded_subgoal="beta",
        )
        _add_timestep(
            episode0,
            4,
            demonstration=False,
            robot_endeffector_p=[10, 11, 12],
            robot_endeffector_q=[1.0, 0.0, 0.0, 0.0],
        )


def test_list_episode_indices_resolves_directory_and_file_path(tmp_path):
    env_id = "TestEnv"
    file_path = tmp_path / f"record_dataset_{env_id}.h5"
    _build_test_h5(file_path, env_id=env_id)

    assert list_episode_indices(env_id, tmp_path) == [0, 2]
    assert list_episode_indices(env_id, file_path) == [0, 2]


def test_get_step_joint_angle_and_non_demo_indexing(tmp_path):
    env_id = "TestEnv"
    file_path = tmp_path / f"record_dataset_{env_id}.h5"
    _build_test_h5(file_path, env_id=env_id)

    with EpisodeDatasetResolver(env_id=env_id, episode=0, dataset_directory=tmp_path) as resolver:
        np.testing.assert_allclose(
            resolver.get_step("joint_angle", 0),
            np.array([1, 2, 3, 4, 5, 6, 7, -1], dtype=np.float64),
        )
        np.testing.assert_allclose(
            resolver.get_step("joint_angle", 1),
            np.array([8, 9, -1, -1, -1, -1, -1, -1], dtype=np.float64),
        )
        assert resolver.get_step("joint_angle", 2) is None
        assert resolver.get_step("joint_angle", 3) is None
        assert resolver.get_step("joint_angle", 4) is None


def test_get_step_keypoint_filters_non_demo_and_requires_keypoint_fields(tmp_path):
    env_id = "TestEnv"
    file_path = tmp_path / f"record_dataset_{env_id}.h5"
    _build_test_h5(file_path, env_id=env_id)

    with EpisodeDatasetResolver(env_id=env_id, episode=0, dataset_directory=file_path) as resolver:
        np.testing.assert_allclose(
            resolver.get_step("keypoint", 0),
            np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            resolver.get_step("keypoint", 1),
            np.array([0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float64),
        )
        assert resolver.get_step("keypoint", 2) is None


def test_get_step_ee_pose_gripper_returns_none_on_missing_pose_fields(tmp_path):
    env_id = "TestEnv"
    file_path = tmp_path / f"record_dataset_{env_id}.h5"
    _build_test_h5(file_path, env_id=env_id)

    with EpisodeDatasetResolver(env_id=env_id, episode=0, dataset_directory=tmp_path) as resolver:
        np.testing.assert_allclose(
            resolver.get_step("ee_pose", 0),
            np.array([1, 2, 3, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=np.float64),
        )
        assert resolver.get_step("ee_pose", 1) is None
        np.testing.assert_allclose(
            resolver.get_step("ee_pose", 2),
            np.array([7, 8, 9, 0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            resolver.get_step("ee_pose", 3),
            np.array([10, 11, 12, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64),
        )


def test_get_step_grounded_subgoal_distinct_sequence_and_invalid_inputs(tmp_path):
    env_id = "TestEnv"
    file_path = tmp_path / f"record_dataset_{env_id}.h5"
    _build_test_h5(file_path, env_id=env_id)

    with EpisodeDatasetResolver(env_id=env_id, episode=0, dataset_directory=tmp_path) as resolver:
        assert resolver.get_step("grounded_subgoal", 0) == "alpha"
        assert resolver.get_step("grounded_subgoal", 1) == "beta"
        assert resolver.get_step("grounded_subgoal", 2) is None
        assert resolver.get_step("grounded_subgoal", 3) is None
        assert resolver.get_step("joint_angle", -1) is None
        assert resolver.get_step("invalid_mode", 0) is None  # type: ignore[arg-type]
