"""
Episode dataset resolver: read h5 per-episode timestep data (actions, demonstration flag).
Similar to EpisodeConfigResolver for metadata; this class reads dataset content from h5.
"""
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

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
    Resolves per-timestep dataset content (action, demonstration) for one (env_id, episode)
    from h5. Open h5 on init; call get_action(step) for replay step (0=first non-demo), or
    get_action_by_stored_index(index) / get_action_from_absolute_timestep(step). close() or use as context manager when done.
    """

    def __init__(
        self,
        env_id: str,
        episode: int,
        dataset_directory: Optional[Union[str, Path]] = None,
        *,
        dataset_path: Optional[Union[str, Path]] = None,
    ):
        self.env_id = env_id
        self.episode = episode

        # Backward-compatible alias: old callsites may still pass dataset_path.
        if dataset_directory is None:
            dataset_directory = dataset_path
        elif dataset_path is not None and Path(dataset_directory) != Path(dataset_path):
            raise ValueError("Provide only one of dataset_directory or dataset_path.")
        if dataset_directory is None:
            raise ValueError("dataset_directory is required.")

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
        self._non_demo_record_steps: List[int] = []
        self._scan_cursor = 0
        self._non_demo_keypoint_record_steps: List[int] = []
        self._keypoint_scan_cursor = 0
        self._distinct_subgoal_record_steps: List[int] = []
        self._distinct_subgoal_built: bool = False

    def _read_subgoal_at_record_step(self, record_step: int) -> str:
        """
        Read grounded_subgoal (or subgoal fallback) at the given record_timestep.
        Returns normalized string; empty string if missing/None.
        """
        key = f"record_timestep_{record_step}"
        if key not in self._episode_group:
            return ""
        timestep_group = self._episode_group[key]
        val = None
        if "grounded_subgoal" in timestep_group:
            val = timestep_group["grounded_subgoal"][()]
        elif "subgoal" in timestep_group:
            val = timestep_group["subgoal"][()]
        if val is None:
            return ""
        s = val.decode("utf-8") if hasattr(val, "decode") else str(val)
        return s if s is not None else ""

    def _ensure_distinct_subgoals(self) -> None:
        """
        Build _distinct_subgoal_record_steps: record_steps (non-demo only) where
        grounded_subgoal text changes from the previous non-demo step. First non-demo step is always included.
        """
        if self._distinct_subgoal_built:
            return
        prev_subgoal: Optional[str] = None
        for record_step in self._timestep_indexes:
            key = f"record_timestep_{record_step}"
            if key not in self._episode_group:
                continue
            timestep_group = self._episode_group[key]
            is_demo = False
            if "demonstration" in timestep_group:
                is_demo = _as_bool(timestep_group["demonstration"][()])
            if is_demo:
                continue
            current = self._read_subgoal_at_record_step(record_step)
            if prev_subgoal is None or current != prev_subgoal:
                self._distinct_subgoal_record_steps.append(record_step)
            prev_subgoal = current
        self._distinct_subgoal_built = True

    def get_keypoint(self, step: int) -> Optional[np.ndarray]:
        """
        Return 8-d keypoint action [keypoint_p(3), keypoint_q(4), gripper(1)]
        for the step-th non-demonstration timestep that has keypoint data
        (keypoint_p, keypoint_q), for all envs.
        Returns None when step is out of range or no more keypoints.
        """
        while step >= len(self._non_demo_keypoint_record_steps) and self._keypoint_scan_cursor < len(self._timestep_indexes):
            record_step = self._timestep_indexes[self._keypoint_scan_cursor]
            key = f"record_timestep_{record_step}"
            timestep_group = self._episode_group[key]
            is_demo = False
            if "demonstration" in timestep_group:
                is_demo = _as_bool(timestep_group["demonstration"][()])
            has_keypoint = "keypoint_p" in timestep_group and "keypoint_q" in timestep_group
            self._keypoint_scan_cursor += 1
            if not is_demo and has_keypoint:
                self._non_demo_keypoint_record_steps.append(record_step)
        if step >= len(self._non_demo_keypoint_record_steps):
            return None
        record_step = self._non_demo_keypoint_record_steps[step]
        key = f"record_timestep_{record_step}"
        if key not in self._episode_group:
            return None
        timestep_group = self._episode_group[key]
        p = timestep_group["keypoint_p"][()] if "keypoint_p" in timestep_group else None
        q = timestep_group["keypoint_q"][()] if "keypoint_q" in timestep_group else None
        if p is None or q is None:
            return None
        raw_action = timestep_group["action"][()] if "action" in timestep_group else None
        action_8d = _action_to_8d(raw_action)
        gripper = float(action_8d[-1]) if action_8d is not None and len(action_8d) > 0 else -1.0
        return np.concatenate([
            np.asarray(p, dtype=np.float64).flatten()[:3],
            np.asarray(q, dtype=np.float64).flatten()[:4],
            [gripper],
        ]).astype(np.float64)

    def get_action(self, step: int) -> Optional[np.ndarray]:
        """
        Return 8-d joint action for the step-th non-demo timestep (0 = first non-demo),
        for all envs.
        Builds non-demo index lazily: only scans forward as needed, no full iterate.
        Returns None when step is out of range (no more steps).
        """
        while step >= len(self._non_demo_record_steps) and self._scan_cursor < len(self._timestep_indexes):
            record_step = self._timestep_indexes[self._scan_cursor]
            action, is_demo = self.get_action_from_absolute_timestep(record_step)
            self._scan_cursor += 1
            if not is_demo:
                self._non_demo_record_steps.append(record_step)
        if step < len(self._non_demo_record_steps):
            record_step = self._non_demo_record_steps[step]
            action, _ = self.get_action_from_absolute_timestep(record_step)
            return action
        return None

    def get_ee_pose(self, step: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (p, q) for the step-th non-demo timestep.
        """
        while step >= len(self._non_demo_record_steps) and self._scan_cursor < len(self._timestep_indexes):
            record_step = self._timestep_indexes[self._scan_cursor]
            key = f"record_timestep_{record_step}"
            timestep_group = self._episode_group[key]
            is_demo = False
            if "demonstration" in timestep_group:
                is_demo = _as_bool(timestep_group["demonstration"][()])
            
            self._scan_cursor += 1
            if not is_demo:
                self._non_demo_record_steps.append(record_step)
        
        if step < len(self._non_demo_record_steps):
            record_step = self._non_demo_record_steps[step]
            return self.get_ee_pose_from_absolute_timestep(record_step)
        return None, None

    def get_ee_pose_gripper(self, step: int) -> Optional[np.ndarray]:
        """
        Return 8-d ee action [ee_p(3), ee_q(4), gripper(1)] for the step-th
        non-demo timestep, for all envs.
        Same step resolution as get_ee_pose. Returns None if pose is missing.
        """
        while step >= len(self._non_demo_record_steps) and self._scan_cursor < len(self._timestep_indexes):
            record_step = self._timestep_indexes[self._scan_cursor]
            key = f"record_timestep_{record_step}"
            timestep_group = self._episode_group[key]
            is_demo = False
            if "demonstration" in timestep_group:
                is_demo = _as_bool(timestep_group["demonstration"][()])
            self._scan_cursor += 1
            if not is_demo:
                self._non_demo_record_steps.append(record_step)
        if step >= len(self._non_demo_record_steps):
            return None
        record_step = self._non_demo_record_steps[step]
        key = f"record_timestep_{record_step}"
        if key not in self._episode_group:
            return None
        timestep_group = self._episode_group[key]
        p = timestep_group["robot_endeffector_p"][()] if "robot_endeffector_p" in timestep_group else None
        q = timestep_group["robot_endeffector_q"][()] if "robot_endeffector_q" in timestep_group else None
        if p is None or q is None:
            return None
        p_flat = np.asarray(p, dtype=np.float64).flatten()[:3]
        q_flat = np.asarray(q, dtype=np.float64).flatten()[:4]
        raw_action = timestep_group["action"][()] if "action" in timestep_group else None
        action_8d = _action_to_8d(raw_action)
        gripper = float(action_8d[-1]) if action_8d is not None and len(action_8d) > 0 else -1.0
        return np.concatenate([p_flat, q_flat, [gripper]]).astype(np.float64)

    def get_grounded_subgoal(self, step_idx: int) -> Optional[str]:
        """
        Return the grounded subgoal text for the step_idx-th *distinct* subgoal.
        Distinct = when the subgoal text changes from the previous (non-demo) timestep.
        step_idx 0 = first subgoal, 1 = first change, etc. Returns None if step_idx out of range.
        """
        self._ensure_distinct_subgoals()
        if step_idx < 0 or step_idx >= len(self._distinct_subgoal_record_steps):
            return None
        record_step = self._distinct_subgoal_record_steps[step_idx]
        text = self._read_subgoal_at_record_step(record_step)
        return text if text else None

    def get_ee_pose_from_absolute_timestep(self, step: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key = f"record_timestep_{step}"
        if key not in self._episode_group:
            return None, None
        timestep_group = self._episode_group[key]
        p = timestep_group["robot_endeffector_p"][()] if "robot_endeffector_p" in timestep_group else None
        q = timestep_group["robot_endeffector_q"][()] if "robot_endeffector_q" in timestep_group else None
        return p, q

    def get_action_from_absolute_timestep(self, step: int) -> Tuple[Optional[np.ndarray], bool]:
        """
        Return (action_8d, is_demonstration) for the given record_timestep number.
        If action is missing/None/"None", returns (None, is_demo).
        """
        key = f"record_timestep_{step}"
        if key not in self._episode_group:
            return None, False
        timestep_group = self._episode_group[key]
        is_demo = False
        if "demonstration" in timestep_group:
            is_demo = _as_bool(timestep_group["demonstration"][()])
        raw_action = timestep_group["action"][()] if "action" in timestep_group else None
        action = _action_to_8d(raw_action)
        return action, is_demo

    def get_action_by_stored_index(self, index: int) -> Tuple[Optional[np.ndarray], bool]:
        """
        Return (action_8d, is_demonstration) for the index-th timestep in stored order.
        Maps sequential index (0, 1, 2, ...) to the corresponding record_timestep.
        If index is out of range, returns (None, False).
        """
        if index < 0 or index >= len(self._timestep_indexes):
            return None, False
        record_step = self._timestep_indexes[index]
        return self.get_action_from_absolute_timestep(record_step)



    def close(self) -> None:
        """Close the h5 file. Idempotent."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "EpisodeDatasetResolver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
