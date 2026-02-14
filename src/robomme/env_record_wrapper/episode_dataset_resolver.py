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
    read episode_{N} keys from root (or env_{env_id} for backward compatibility), 
    return sorted episode numbers. Raises if file missing.
    """
    h5_path = _resolve_h5_path(env_id, dataset_directory)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    with h5py.File(h5_path, "r") as h5:
        # Check for env_{env_id} group (legacy format)
        env_group_key = f"env_{env_id}"
        if env_group_key in h5:
            source_group = h5[env_group_key]
        else:
            # Assume root level episodes (new format)
            source_group = h5

        indices = sorted(
            int(k.split("_")[1])
            for k in source_group.keys()
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
    action = np.asarray(raw_action).flatten()
    if action.size == 0:
        return None
    if action.size == 7:
        action = np.concatenate([action, [-1.0]])
    elif action.size < 8:
        action = np.pad(action, (0, 8 - action.size), constant_values=-1.0)
    return action[:8]


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
        
        # Try finding episode group in root (new format) or env_ group (old format)
        episode_key = f"episode_{episode}"
        env_group_key = f"env_{env_id}"
        
        if episode_key in self._h5:
             self._episode_group = self._h5[episode_key]
        elif env_group_key in self._h5:
            env_group = self._h5[env_group_key]
            if episode_key not in env_group:
                self._h5.close()
                raise KeyError(f"H5 missing group '{episode_key}' in '{env_group_key}' of {self._h5_path}")
            self._episode_group = env_group[episode_key]
        else:
            self._h5.close()
            raise KeyError(f"H5 missing group '{episode_key}' (checked root and '{env_group_key}') in {self._h5_path}")

        # Support both 'timestep_N' (new) and 'record_timestep_N' (old)
        self._timestep_indexes = []
        for k in self._episode_group.keys():
            # Check for new format "timestep_N"
            m_new = re.match(r"timestep_(\d+)$", k)
            if m_new:
                self._timestep_indexes.append(int(m_new.group(1)))
                continue
            
            # Check for old format "record_timestep_N"
            m_old = re.match(r"record_timestep_(\d+)$", k)
            if m_old:
                self._timestep_indexes.append(int(m_old.group(1)))
        
        self._timestep_indexes.sort()
        self._timestep_group_cache: Dict[int, h5py.Group] = {}
        self._non_demo_steps: List[int] = []
        self._keypoint_steps: List[int] = []
        self._distinct_subgoal_steps: List[int] = []
        self._build_indexes()

    def _get_timestep_group(self, record_step: int) -> Optional[h5py.Group]:
        if record_step in self._timestep_group_cache:
            return self._timestep_group_cache[record_step]
        
        # Try new format first
        key = f"timestep_{record_step}"
        if key not in self._episode_group:
            # Fallback to old format
            key = f"record_timestep_{record_step}"
            if key not in self._episode_group:
                return None
                
        timestep_group = self._episode_group[key]
        self._timestep_group_cache[record_step] = timestep_group
        return timestep_group

    def _is_video_demo_group(self, timestep_group: h5py.Group) -> bool:
        # New structure: info/is_video_demo
        info_grp = timestep_group.get("info")
        if info_grp is None or "is_video_demo" not in info_grp:
            return False
        return _as_bool(info_grp["is_video_demo"][()])

    def _extract_joint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # New structure: action/joint_action; compatible with old structure: action (direct dataset)
        action_grp = timestep_group.get("action")
        if action_grp is not None and isinstance(action_grp, h5py.Group) and "joint_action" in action_grp:
            raw_action = action_grp["joint_action"][()]
        elif "action" in timestep_group:
            raw_action = timestep_group["action"][()]
        else:
            raw_action = None
        return _action_to_8d(raw_action)

    def _extract_ee_pose_gripper(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # Directly read action/eef_action 7-dim dataset [pose(3), rpy(3), gripper(1)]
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group):
            return None
        if "eef_action" not in action_grp:
            return None
        return np.asarray(action_grp["eef_action"][()]).flatten()

    def _extract_ee_quat_gripper(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # Read action/eef_action_raw/{pose,quat} + action/eef_action[-1] => 8D [pose(3), quat(4), gripper(1)]
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group):
            return None
        if "eef_action_raw" not in action_grp:
            return None

        raw_grp = action_grp["eef_action_raw"]
        if "pose" not in raw_grp or "quat" not in raw_grp:
            return None

        pose = np.asarray(raw_grp["pose"][()]).flatten()[:3]
        quat = np.asarray(raw_grp["quat"][()]).flatten()[:4]
        if pose.size < 3 or quat.size < 4:
            return None

        gripper = -1.0
        if "eef_action" in action_grp:
            try:
                eef_action = np.asarray(action_grp["eef_action"][()]).flatten()
            except (TypeError, ValueError):
                eef_action = np.asarray([])
            if eef_action.size > 0 and np.isfinite(eef_action[-1]):
                gripper = float(eef_action[-1])

        return np.concatenate([pose, quat, [gripper]])

    def _extract_keypoint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # New structure: action/keypoint_action (7D: pos(3)+rpy(3)+gripper(1))
        action_grp = timestep_group.get("action")
        if action_grp is not None and isinstance(action_grp, h5py.Group):
            src = action_grp
        else:
            src = timestep_group
        if "keypoint_action" not in src:
            return None
        return np.asarray(src["keypoint_action"][()]).flatten()

    def _extract_subgoal_text(self, timestep_group: h5py.Group) -> Optional[str]:
        val = None
        # New structure: info/grounded_subgoal; compatible with old structure
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
            if timestep_group is None or self._is_video_demo_group(timestep_group):
                continue

            self._non_demo_steps.append(record_step)
            # is_keyframe: Determine if it is a keypoint refresh frame via info/is_keyframe flag
            info_grp = timestep_group.get("info")
            if info_grp is not None and "is_keyframe" in info_grp:
                if _as_bool(info_grp["is_keyframe"][()]):
                    self._keypoint_steps.append(record_step)

            current_subgoal = self._extract_subgoal_text(timestep_group)
            if prev_subgoal is None or current_subgoal != prev_subgoal:
                self._distinct_subgoal_steps.append(record_step)
            prev_subgoal = current_subgoal

    def get_step(
        self,
        mode: Literal["joint_angle", "ee_pose", "ee_quat", "keypoint", "oracle_planner"],
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
        elif mode == "ee_quat":
            selected_steps = self._non_demo_steps
            extractor = self._extract_ee_quat_gripper
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
