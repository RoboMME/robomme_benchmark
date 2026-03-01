"""
Episode dataset resolver: read h5 per-episode timestep data (actions, demonstration flag).
Similar to EpisodeConfigResolver for metadata; this class reads dataset content from h5.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
    read episode_{N} keys from root, return sorted episode numbers. Raises if file missing.
    """
    h5_path = _resolve_h5_path(env_id, dataset_directory)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    with h5py.File(h5_path, "r") as h5:
        indices = sorted(
            int(k.split("_")[1])
            for k in h5.keys()
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
    Build non-demo / waypoint / oracle-command indexes at initialization and
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
        
        episode_key = f"episode_{episode}"
        if episode_key not in self._h5:
            self._h5.close()
            raise KeyError(f"H5 missing group '{episode_key}' in {self._h5_path}")
        self._episode_group = self._h5[episode_key]

        self._timestep_indexes = sorted(
            int(m.group(1))
            for k in self._episode_group.keys()
            for m in [re.match(r"timestep_(\d+)$", k)]
            if m
        )
        self._timestep_group_cache: Dict[int, h5py.Group] = {}
        self._non_demo_steps: List[int] = []
        self._waypoint_steps: List[int] = []
        # multi_choice: ordered by timestep where info/is_subgoal_boundary=True
        self._oracle_commands: List[Dict[str, Any]] = []
        self._build_indexes()

    def _get_timestep_group(self, record_step: int) -> Optional[h5py.Group]:
        if record_step in self._timestep_group_cache:
            return self._timestep_group_cache[record_step]
        
        key = f"timestep_{record_step}"
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

    def _is_choice_subgoal_boundary_group(self, timestep_group: h5py.Group) -> bool:
        info_grp = timestep_group.get("info")
        if info_grp is None or "is_subgoal_boundary" not in info_grp:
            return False
        return _as_bool(info_grp["is_subgoal_boundary"][()])

    def _extract_joint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group) or "joint_action" not in action_grp:
            return None
        return _action_to_8d(action_grp["joint_action"][()])

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

    def _extract_waypoint_action(self, timestep_group: h5py.Group) -> Optional[np.ndarray]:
        # action/waypoint_action (7D: pos(3)+rpy(3)+gripper(1))
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group) or "waypoint_action" not in action_grp:
            return None
        try:
            waypoint_action = np.asarray(action_grp["waypoint_action"][()]).flatten()
        except (TypeError, ValueError):
            return None

        if waypoint_action.shape != (7,):
            return None
        if not np.all(np.isfinite(waypoint_action)):
            return None
        return waypoint_action

    @staticmethod
    def _decode_h5_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = np.reshape(value, -1)[0]
        if isinstance(value, (bytes, np.bytes_)):
            try:
                return value.decode("utf-8")
            except Exception:
                return None
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _normalize_choice_point(point_like: Any) -> Optional[List[float]]:
        # New schema: choice_action.point stores front_rgb pixel [y, x].
        if not isinstance(point_like, (list, tuple, np.ndarray)) or len(point_like) != 2:
            return None
        try:
            y = float(point_like[0])
            x = float(point_like[1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return [y, x]

    def _extract_choice_action(self, timestep_group: h5py.Group) -> Optional[Dict[str, Any]]:
        action_grp = timestep_group.get("action")
        if action_grp is None or not isinstance(action_grp, h5py.Group):
            return None
        if "choice_action" not in action_grp:
            return None

        payload_raw = self._decode_h5_string(action_grp["choice_action"][()])
        if not payload_raw:
            return None
        try:
            payload = json.loads(payload_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        choice = payload.get("choice")
        if not isinstance(choice, str):
            return None
        choice = choice.strip().upper()
        if not choice:
            return None
        if "point" not in payload:
            return None

        command: Dict[str, Any] = {"choice": choice}
        point = self._normalize_choice_point(payload.get("point"))
        if point is None:
            return None
        command["point"] = point

        return command

    def _build_indexes(self) -> None:
        # Collect oracle commands in timestep order (only timesteps marked by info/is_subgoal_boundary)
        oracle_commands: List[Dict[str, Any]] = []
        prev_waypoint_action: Optional[np.ndarray] = None
        for record_step in self._timestep_indexes:
            timestep_group = self._get_timestep_group(record_step)
            if timestep_group is None or self._is_video_demo_group(timestep_group):
                continue

            self._non_demo_steps.append(record_step)
            # waypoint_action is stored per step; keep only adjacent changes for logical
            # sparse sequence and skip invalid/non-finite sentinels (do not rely on
            # info/is_subgoal_boundary).
            waypoint_action = self._extract_waypoint_action(timestep_group)
            if waypoint_action is not None:
                if prev_waypoint_action is None or not np.array_equal(
                    waypoint_action, prev_waypoint_action
                ):
                    self._waypoint_steps.append(record_step)
                    prev_waypoint_action = waypoint_action.copy()

            if not self._is_choice_subgoal_boundary_group(timestep_group):
                continue
            command = self._extract_choice_action(timestep_group)
            if command is not None:
                oracle_commands.append(command)

        self._oracle_commands = oracle_commands

    def get_step(
        self,
        mode: Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"],
        step: int,
    ) -> Optional[Union[np.ndarray, Dict[str, Any]]]:
        if step < 0:
            return None

        if mode == "multi_choice":
            if step >= len(self._oracle_commands):
                return None
            command = self._oracle_commands[step]
            return dict(command)

        if mode == "joint_angle":
            selected_steps = self._non_demo_steps
            extractor = self._extract_joint_action
        elif mode == "ee_pose":
            selected_steps = self._non_demo_steps
            extractor = self._extract_ee_pose_gripper

        elif mode == "waypoint":
            selected_steps = self._waypoint_steps
            extractor = self._extract_waypoint_action
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
