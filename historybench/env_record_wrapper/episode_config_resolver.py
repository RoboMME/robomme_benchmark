"""
Episode 配置解析：从元数据解析 episode 的 seed、difficulty，并构建包装好的环境。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym

DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset_json"
_ALLOWED_DATASETS = {"train"}
_ALLOWED_ACTION_SPACES = {"joint_angle", "ee_pose", "ee_quat", "keypoint", "oracle_planner"}
_DEFAULT_TASK_LIST = [
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
]


def load_episode_metadata(metadata_path: Union[str, Path, None]) -> Dict[Tuple[str, int], Dict[str, object]]:
    """
    从 JSON 文件读取每集的元数据（metadata）；如果缺失或无效则返回空字典。
    用于恢复特定 episode 的配置（如 seed、难度等）。
    """

    metadata_index: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not metadata_path:
        return metadata_index

    path = Path(metadata_path)
    if not path.exists():
        print(f"Metadata file not found, skipping: {path}")
        return metadata_index

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - informational logging only
        print(f"Failed to read metadata {path}: {exc}")
        return metadata_index

    default_task = str(payload.get("env_id") or "").strip()
    for record in payload.get("records", []):
        task_name = str(record.get("task") or default_task or "").strip()
        episode = record.get("episode")
        if not task_name or episode is None:
            continue
        try:
            episode_idx = int(episode)
        except (TypeError, ValueError):
            continue
        metadata_index[(task_name, episode_idx)] = record

    if metadata_index:
        print(f"Loaded {len(metadata_index)} metadata records from {path}")
    else:
        print(f"No valid metadata entries found in {path}")
    return metadata_index


def get_episode_metadata(
    metadata_index: Dict[Tuple[str, int], Dict[str, object]],
    task: str,
    episode: int,
) -> Optional[Dict[str, object]]:
    """查找特定 (task, episode) 配对的元数据条目。"""

    if not metadata_index:
        return None
    return metadata_index.get((task, episode))


class BenchmarkEnvBuilder:
    """
    Episode 环境构建器。

    根据 dataset 与 env_id 自动解析 metadata，并按 action_space 构建包装好的环境。
    """

    def __init__(
        self,
        env_id: str,
        dataset: str,
        action_space: str,
        gui_render: bool,
        override_metadata_path: Optional[Union[str, Path]] = None,
    ):
        if dataset not in _ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset}'. Allowed datasets: {sorted(_ALLOWED_DATASETS)}")
        if action_space not in _ALLOWED_ACTION_SPACES:
            raise ValueError(
                f"Unsupported action_space '{action_space}'. "
                f"Allowed action spaces: {sorted(_ALLOWED_ACTION_SPACES)}"
            )

        self.env_id = env_id
        self.dataset = dataset
        self.action_space = action_space
        self.gui_render = gui_render
        self.override_metadata_path = (
            Path(override_metadata_path) if override_metadata_path is not None else None
        )
        self.render_mode = "human" if gui_render else "rgb_array"
        self.max_steps_without_demonstration = 10000

        metadata_path = self._resolve_metadata_path()
        self.metadata_index = load_episode_metadata(metadata_path)

    @classmethod
    def get_task_list(cls) -> List[str]:
        """
        返回可评测任务列表。
        任务列表固定为内置默认顺序，不从 metadata 自动发现。
        """

        return list(_DEFAULT_TASK_LIST)

    def _resolve_metadata_path(self) -> str:
        if self.override_metadata_path is not None:
            return str(
                self.override_metadata_path / f"record_dataset_{self.env_id}_metadata.json"
            )
        if self.dataset == "train":
            return os.path.join(str(DATASET_ROOT), f"record_dataset_{self.env_id}_metadata.json")
        raise ValueError(f"Unsupported dataset '{self.dataset}'.")

    def resolve_episode(self, episode: int):
        """根据 metadata 解析 episode 的配置。"""
        seed = None
        difficulty_hint = None

        metadata = get_episode_metadata(self.metadata_index, self.env_id, episode)
        if metadata:
            metadata_seed = metadata.get("seed")
            if metadata_seed is not None:
                try:
                    seed = int(metadata_seed)
                except (TypeError, ValueError):
                    print(f"[{self.env_id}] Invalid metadata seed for episode {episode}: {metadata_seed}")
            difficulty_hint = metadata.get("difficulty")

        return seed, difficulty_hint

    def get_episode_num(self) -> int:
        """
        返回当前 env_id 在 metadata 中的 episode 数量。
        注意：按当前约定，该方法名返回数量（int）而非列表。
        """
        if not self.metadata_index:
            return 0
        episode_set = {episode for (task, episode) in self.metadata_index if task == self.env_id}
        return len(episode_set)

    def make_env_for_episode(self, episode: int):
        """为特定 episode 创建并配置环境。action_space=ee_pose/ee_quat 时包 EndeffectorDemonstrationWrapper，keypoint 时包 MultiStepDemonstrationWrapper，oracle_planner 时包 OraclePlannerDemonstrationWrapper。"""
        from .DemonstrationWrapper import DemonstrationWrapper

        seed, difficulty_hint = self.resolve_episode(episode)
        env_kwargs = dict(
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode=self.render_mode,
            reward_mode="dense",
            max_episode_steps=99999,
        )
        if seed is not None:
            env_kwargs["HistoryBench_seed"] = seed
        if difficulty_hint:
            env_kwargs["HistoryBench_difficulty"] = difficulty_hint
        seed_desc = seed if seed is not None else "default"
        difficulty_str = f", difficulty={difficulty_hint}" if difficulty_hint else ""
        print(f"[{self.env_id}] Episode {episode}: seed={seed_desc}{difficulty_str}")

        env = gym.make(self.env_id, **env_kwargs)
        env = DemonstrationWrapper(
            env,
            max_steps_without_demonstration=self.max_steps_without_demonstration,
            gui_render=self.gui_render,
        )
        if self.action_space == "joint_angle":
            pass
        elif self.action_space == "ee_pose":
            from .EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper

            env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
        elif self.action_space == "ee_quat":
            from .EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper

            env = EndeffectorDemonstrationWrapper(env, action_repr="quat")
        elif self.action_space == "keypoint":
            from .MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper

            env = MultiStepDemonstrationWrapper(env, gui_render=self.gui_render, vis=self.gui_render)
        elif self.action_space == "oracle_planner":
            from .OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper
            env = OraclePlannerDemonstrationWrapper(env, env_id=self.env_id, gui_render=self.gui_render)

        return env, seed, difficulty_hint
