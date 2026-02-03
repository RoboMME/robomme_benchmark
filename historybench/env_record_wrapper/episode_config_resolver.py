"""
Episode 配置解析：从元数据解析 episode 的 seed、difficulty，并构建包装好的环境。
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym


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


class EpisodeConfigResolver:
    """
    Episode 配置解析器。

    辅助类，用于解析每个 episode 的种子（seed）和难度（difficulty），并构建包装好的环境。
    数据来源为元数据文件。
    """

    def __init__(
        self,
        env_id: str,
        metadata_path: Union[str, Path, None],
        render_mode: str,
        gui_render: bool,
        max_steps_without_demonstration: int,
        save_video: bool = False,
        action_space: Optional[str] = None,
    ):
        self.env_id = env_id
        self.render_mode = render_mode
        self.gui_render = gui_render
        self.max_steps_without_demonstration = max_steps_without_demonstration
        self.save_video = save_video
        self.action_space = action_space
        self.metadata_index = load_episode_metadata(metadata_path)

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

    def make_env_for_episode(self, episode: int):
        """为特定 episode 创建并配置环境。"""
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
            action_space=self.action_space,
        )
        return env, seed, difficulty_hint
