"""
进程内会话管理模块。

当前实现不再为每个用户启动独立 worker 进程，而是在 Gradio 主进程中保存
每个用户的 OracleSession，并将重计算边界统一下沉到 ZeroGPU 适配层。
"""

from __future__ import annotations

import logging
import os
import sys

# 添加父目录到路径（逻辑复制自 oracle_logic.py）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from oracle_logic import DEFAULT_DATASET_ROOT, OracleSession
from zerogpu_runtime import execute_action_gpu, load_episode_gpu, update_observation_gpu

try:
    from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure
except ImportError:
    ScrewPlanFailure = RuntimeError


LOGGER = logging.getLogger("robomme.process_session")


class ScrewPlanFailureError(RuntimeError):
    """Exception raised when screw plan fails, to be caught and displayed via gr.Info popup."""


def _sanitize_options(options):
    """
    清理选项数据，移除不可序列化或不可稳定缓存的项（如 'solve' 函数）。
    """
    clean_opts = []
    if not options:
        return clean_opts
    for opt in options:
        clean_opt = opt.copy()
        if "solve" in clean_opt:
            del clean_opt["solve"]
        if "available" in clean_opt:
            clean_opt["available"] = bool(clean_opt["available"])
        clean_opts.append(clean_opt)
    return clean_opts


class ProcessSessionProxy:
    """
    保留旧的代理类命名，但内部已改为单进程本地会话。
    """

    def __init__(self, dataset_root=DEFAULT_DATASET_ROOT, gui_render=False):
        self._session = OracleSession(dataset_root=dataset_root, gui_render=gui_render)
        self._sync_state(reset_execution_frames=True)
        LOGGER.info(
            "ProcessSessionProxy initialized in-process dataset_root=%s gui_render=%s",
            dataset_root,
            gui_render,
        )

    def _sync_state(self, *, reset_execution_frames=False):
        session = self._session
        self.env_id = session.env_id
        self.episode_idx = session.episode_idx
        self.language_goal = session.language_goal
        self.difficulty = session.difficulty
        self.seed = session.seed
        self.demonstration_frames = session.demonstration_frames
        self.base_frames = session.base_frames
        self.wrist_frames = session.wrist_frames
        self.available_options = session.available_options
        self.raw_solve_options = _sanitize_options(session.raw_solve_options)
        self.seg_vis = session.seg_vis
        self.is_demonstration = bool(
            getattr(session.env, "current_task_demonstration", False) if session.env else False
        )
        self.non_demonstration_task_length = session.non_demonstration_task_length
        if reset_execution_frames:
            self.last_execution_frames = []

    def load_episode(self, env_id, episode_idx):
        result = load_episode_gpu(self._session, env_id, episode_idx)
        self._sync_state(reset_execution_frames=True)
        return result

    def execute_action(self, action_idx, click_coords):
        self.last_execution_frames = []
        execute_base_start = len(self._session.base_frames)
        try:
            result = execute_action_gpu(self._session, action_idx, click_coords)
        except ScrewPlanFailure as exc:
            self._sync_state(reset_execution_frames=False)
            raise ScrewPlanFailureError(f"screw plan failed: {exc}") from exc
        except Exception:
            self._sync_state(reset_execution_frames=False)
            raise
        self._sync_state(reset_execution_frames=False)
        self.last_execution_frames = self._session.base_frames[execute_base_start:]
        return result

    def get_pil_image(self, use_segmented=True):
        return self._session.get_pil_image(use_segmented=use_segmented)

    def update_observation(self, use_segmentation=True):
        result = update_observation_gpu(self._session, use_segmentation)
        self._sync_state(reset_execution_frames=False)
        return result

    def get_reference_action(self):
        result = self._session.get_reference_action()
        self._sync_state(reset_execution_frames=False)
        return result

    def close(self):
        self._session.close()
        LOGGER.info("ProcessSessionProxy closed in-process env=%s episode=%s", self.env_id, self.episode_idx)
