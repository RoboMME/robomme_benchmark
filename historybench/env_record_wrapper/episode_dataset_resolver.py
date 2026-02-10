"""
Episode dataset resolver: read h5 per-episode timestep data (actions, demonstration flag).
Similar to EpisodeConfigResolver for metadata; this class reads dataset content from h5.
"""
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import h5py
import numpy as np


def _resolve_h5_path(env_id: str, dataset_directory: Union[str, Path]) -> Path:
    """
    Resolve h5 file path: if dataset_directory is a full path to a .h5 file (suffix .h5), use it;
    otherwise treat as directory and use dataset_directory/record_dataset_{env_id}.h5.
    """
    p = Path(dataset_directory)
    if p.suffix == ".h5":
        return p
    return p / f"record_dataset_{env_id}.h5"


def list_episode_indices(env_id: str, dataset_directory: Union[str, Path]) -> List[int]:
    """
    Open h5 at dataset_directory (full path to .h5) or dataset_directory/record_dataset_{env_id}.h5,
    read env_{env_id} keys, return sorted episode numbers, then close. Raises if file or group missing.
    """
    h5_path = _resolve_h5_path(env_id, dataset_directory)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        env_group_key = f"env_{env_id}"
        if env_group_key not in h5:
            raise KeyError(f"H5 missing group '{env_group_key}' in {h5_path}")
        env_group = h5[env_group_key]
        indices = sorted(
            int(k.split("_")[1])
            for k in env_group.keys()
            if k.startswith("episode_") and re.match(r"episode_\d+", k)
        )
    return indices


def _as_bool(value) -> bool:
    """Convert h5 scalar / array / bytes / None to bool (no torch dependency)."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(np.reshape(value, -1)[0].item())
    if hasattr(value, "decode"):
        value = value.decode("utf-8") if isinstance(value, bytes) else value
    return bool(value) if value is not None else False


def _action_to_8d(raw_action) -> Optional[np.ndarray]:
    """
    Normalize raw h5 action (scalar, array, string "None") to 8d numpy.
    Returns None if action is missing/None/"None".
    """
    if raw_action is None:
        return None
    if hasattr(raw_action, "decode"):
        raw_action = raw_action.decode("utf-8") if isinstance(raw_action, bytes) else raw_action
    if isinstance(raw_action, str) and raw_action.strip().lower() == "none":
        return None
    action = np.asarray(raw_action, dtype=np.float64).flatten()
    if action.size == 0:
        return None
    if action.size == 7:
        action = np.concatenate([action, [-1.0]])
    elif action.size < 8:
        action = np.pad(action, (0, 8 - action.size), constant_values=-1.0)
    return action[:8].astype(np.float64)


class EpisodeDatasetResolver:
    """
    Resolves per-timestep dataset content for one (env_id, episode) from h5.
    Build non-demo / keypoint / distinct-subgoal indexes at initialization and
    query via get_step(mode, step).
    """

    def __init__(
        self,
        env_id: str,
        episode: int,
        dataset_directory: Union[str, Path],
    ):
        self.env_id = env_id
        self.episode = episode

        self._h5_path = _resolve_h5_path(env_id, dataset_directory)
        if not self._h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self._h5_path}")
        self._h5 = h5py.File(self._h5_path, "r")
        env_group_key = f"env_{env_id}"
        if env_group_key not in self._h5:
            self._h5.close()
            raise KeyError(f"H5 missing group '{env_group_key}' in {self._h5_path}")
        env_group = self._h5[env_group_key]
        episode_key = f"episode_{episode}"
        if episode_key not in env_group:
            self._h5.close()
            raise KeyError(f"H5 missing group '{episode_key}' in {self._h5_path}")
        self._episode_group = env_group[episode_key]
        self._timestep_indexes = sorted(
            int(m.group(1))
            for k in self._episode_group.keys()
            if (m := re.match(r"record_timestep_(\d+)$", k))
        )
        self._timestep_group_cache: Dict[int, h5py.Group] = {}
        self._non_demo_steps: List[int] = []
        self._keypoint_steps: List[int] = []
        self._distinct_subgoal_steps: List[int] = []
        self._build_indexes()

    def _get_timestep_group(self, record_step: int) -> Optional[h5py.Group]:
        if record_step in self._timestep_group_cache:
            return self._timestep_group_cache[record_step]
        key = f"record_timestep_{record_step}"
        if key not in self._episode_group:
            return None
        timestep_group = self._episode_group[key]
        self._timestep_group_cache[record_step] = timestep_group
        return timestep_group

    def _is_demo_group(self, timestep_group: h5py.Group) -> bool:
        # 新结构: info/is_demo；兼容旧结构: demonstration
        info_grp = timestep_group.get("info")
        if info_grp is not None and "is_demo" in info_grp:
            return _as_bool(info_grp["is_demo"][()])
        if "demonstration" not in timestep_group:
            return False
        return _as_bool(timestep_group["demonstration"][()])

    def _extract_joint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # 新结构: action/joint_action；兼容旧结构: action（直接 dataset）
        action_grp = timestep_group.get("action")
        if action_grp is not None and isinstance(action_grp, h5py.Group) and "joint_action" in action_grp:
            raw_action = action_grp["joint_action"][()]
        elif "action" in timestep_group:
            raw_action = timestep_group["action"][()]
        else:
            raw_action = None
        return _action_to_8d(raw_action)

    def _extract_ee_pose_gripper(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # 从新 HDF5 结构读取：action/eef_action/pose(3) + action/eef_action/rpy(3)
        # 输出 [pose(3), rpy(3), gripper(1)] = 7 维
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group):
            return None
        eef_grp = action_grp.get("eef_action")
        if eef_grp is None or not isinstance(eef_grp, h5py.Group):
            return None
        if "pose" not in eef_grp or "rpy" not in eef_grp:
            return None
        pose = np.asarray(eef_grp["pose"][()], dtype=np.float64).flatten()
        rpy = np.asarray(eef_grp["rpy"][()], dtype=np.float64).flatten()
        if pose.size < 3 or rpy.size < 3:
            return None
        # gripper 从 joint_action 最后一位获取
        with np.printoptions(suppress=True):
            print(np.degrees(rpy))
        action_8d = self._extract_joint_action(timestep_group)
        gripper = float(action_8d[-1]) if action_8d is not None and action_8d.size > 0 else -1.0
        return np.concatenate([pose[:3], rpy[:3], [gripper]]).astype(np.float64)

    def _extract_keypoint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # 新结构: action/keypoint_p, action/keypoint_q；兼容旧结构
        action_grp = timestep_group.get("action")
        if action_grp is not None and isinstance(action_grp, h5py.Group):
            src = action_grp
        else:
            src = timestep_group
        p = src["keypoint_p"][()] if "keypoint_p" in src else None
        q = src["keypoint_q"][()] if "keypoint_q" in src else None
        if p is None or q is None:
            return None
        p_flat = np.asarray(p, dtype=np.float64).flatten()
        q_flat = np.asarray(q, dtype=np.float64).flatten()
        if p_flat.size < 3 or q_flat.size < 4:
            return None
        action_8d = self._extract_joint_action(timestep_group)
        gripper = float(action_8d[-1]) if action_8d is not None and action_8d.size > 0 else -1.0
        return np.concatenate([p_flat[:3], q_flat[:4], [gripper]]).astype(np.float64)

    def _extract_subgoal_text(self, timestep_group: h5py.Group) -> Optional[str]:
        val = None
        # 新结构: info/grounded_subgoal；兼容旧结构
        info_grp = timestep_group.get("info")
        if info_grp is not None and isinstance(info_grp, h5py.Group):
            if "grounded_subgoal" in info_grp:
                val = info_grp["grounded_subgoal"][()]
            elif "simple_subgoal" in info_grp:
                val = info_grp["simple_subgoal"][()]
        if val is None and "grounded_subgoal" in timestep_group:
            val = timestep_group["grounded_subgoal"][()]
        if val is None and "subgoal" in timestep_group:
            val = timestep_group["subgoal"][()]
        if val is None:
            return None
        text = val.decode("utf-8") if hasattr(val, "decode") else str(val)
        return text if text else None

    def _build_indexes(self) -> None:
        prev_subgoal: Optional[str] = None
        for record_step in self._timestep_indexes:
            timestep_group = self._get_timestep_group(record_step)
            if timestep_group is None or self._is_demo_group(timestep_group):
                continue

            self._non_demo_steps.append(record_step)
            # 新结构: action/keypoint_p, action/keypoint_q；兼容旧结构
            action_grp = timestep_group.get("action")
            kp_src = action_grp if (action_grp is not None and isinstance(action_grp, h5py.Group)) else timestep_group
            if "keypoint_p" in kp_src and "keypoint_q" in kp_src:
                self._keypoint_steps.append(record_step)

            current_subgoal = self._extract_subgoal_text(timestep_group)
            if prev_subgoal is None or current_subgoal != prev_subgoal:
                self._distinct_subgoal_steps.append(record_step)
            prev_subgoal = current_subgoal

    def get_step(
        self,
        mode: Literal["joint_angle", "ee_pose", "keypoint", "oracle_planner"],
        step: int,
    ) -> Optional[Union[np.ndarray, str]]:
        if step < 0:
            return None

        if mode == "oracle_planner":
            if step >= len(self._distinct_subgoal_steps):
                return None
            timestep_group = self._get_timestep_group(self._distinct_subgoal_steps[step])
            if timestep_group is None:
                return None
            return self._extract_subgoal_text(timestep_group)

        if mode == "joint_angle":
            selected_steps = self._non_demo_steps
            extractor = self._extract_joint_action
        elif mode == "ee_pose":
            selected_steps = self._non_demo_steps
            extractor = self._extract_ee_pose_gripper
        elif mode == "keypoint":
            selected_steps = self._keypoint_steps
            extractor = self._extract_keypoint_action
        else:
            return None

        if step >= len(selected_steps):
            return None
        timestep_group = self._get_timestep_group(selected_steps[step])
        if timestep_group is None:
            return None
        return extractor(timestep_group)

    def close(self) -> None:
        """Close the h5 file. Idempotent."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._timestep_group_cache.clear()

    def __enter__(self) -> "EpisodeDatasetResolver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
