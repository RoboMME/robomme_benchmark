"""
进程内会话管理模块。

当前实现不再为每个用户启动独立 worker 进程，而是在 Gradio 主进程中保存
每个用户的 OracleSession，并将重计算边界统一下沉到 ZeroGPU 适配层。
"""

from __future__ import annotations

import logging
import os
import sys

import cv2
import numpy as np
from PIL import Image

# 添加父目录到路径（逻辑复制自 oracle_logic.py）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from oracle_logic import DEFAULT_DATASET_ROOT, OracleSession
from software_render_session import (
    RemoteSessionError,
    SoftwareRenderSessionClient,
    SoftwareRenderUnsupportedError,
)
from zerogpu_runtime import execute_action_gpu, load_episode_gpu, update_observation_gpu

try:
    from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure
except ImportError:
    ScrewPlanFailure = RuntimeError


LOGGER = logging.getLogger("robomme.process_session")
_ZERO_GPU_UNSUPPORTED_MESSAGE = (
    "Current Hugging Face ZeroGPU Space only provides compute access, and SAPIEN software "
    "rendering is unavailable. Please use a standard GPU Space."
)


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


def _is_spaces_runtime() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


def _gpu_graphics_capability_available() -> bool:
    capabilities_raw = str(os.environ.get("NVIDIA_DRIVER_CAPABILITIES") or "").strip()
    if not capabilities_raw:
        return True
    capabilities = {token.strip().lower() for token in capabilities_raw.split(",") if token.strip()}
    return "graphics" in capabilities or "all" in capabilities


def _should_use_software_render_subprocess() -> bool:
    return _is_spaces_runtime() and not _gpu_graphics_capability_available()


def _format_software_render_error(exc: BaseException) -> str:
    detail = str(exc).strip()
    if not detail:
        return _ZERO_GPU_UNSUPPORTED_MESSAGE
    if _ZERO_GPU_UNSUPPORTED_MESSAGE in detail:
        return detail
    return f"{_ZERO_GPU_UNSUPPORTED_MESSAGE} Details: {detail}"


def _build_default_snapshot():
    return {
        "env_id": None,
        "episode_idx": None,
        "language_goal": "",
        "difficulty": None,
        "seed": None,
        "demonstration_frames": [],
        "base_frames": [],
        "wrist_frames": [],
        "available_options": [],
        "raw_solve_options": [],
        "seg_vis": None,
        "is_demonstration": False,
        "non_demonstration_task_length": None,
        "last_execution_frames": [],
    }


def _snapshot_to_pil_image(snapshot, *, use_segmented=True):
    if use_segmented:
        seg_vis = snapshot.get("seg_vis")
        if seg_vis is None:
            return Image.new("RGB", (255, 255), color="gray")
        rgb = cv2.cvtColor(np.asarray(seg_vis), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    base_frames = snapshot.get("base_frames") or []
    if not base_frames:
        return Image.new("RGB", (255, 255), color="gray")
    frame = np.asarray(base_frames[-1])
    if frame.dtype != np.uint8:
        max_val = float(np.max(frame)) if frame.size else 0.0
        if max_val <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return Image.fromarray(frame)


class ProcessSessionProxy:
    """
    保留旧的代理类命名，但内部已改为单进程本地会话。
    """

    def __init__(self, dataset_root=DEFAULT_DATASET_ROOT, gui_render=False):
        self._backend_mode = (
            "software_render_subprocess"
            if _should_use_software_render_subprocess()
            else "in_process"
        )
        self._session = None
        self._software_session = None
        self._snapshot = _build_default_snapshot()

        if self._backend_mode == "software_render_subprocess":
            self._software_session = SoftwareRenderSessionClient(
                dataset_root=dataset_root,
                gui_render=gui_render,
            )
            self._apply_snapshot(self._snapshot, reset_execution_frames=True)
        else:
            self._session = OracleSession(dataset_root=dataset_root, gui_render=gui_render)
            self._sync_state(reset_execution_frames=True)
        LOGGER.info(
            "ProcessSessionProxy initialized backend_mode=%s dataset_root=%s gui_render=%s",
            self._backend_mode,
            dataset_root,
            gui_render,
        )

    def _apply_snapshot(self, snapshot, *, reset_execution_frames=False):
        snapshot = snapshot or _build_default_snapshot()
        self._snapshot = snapshot
        self.env_id = snapshot.get("env_id")
        self.episode_idx = snapshot.get("episode_idx")
        self.language_goal = snapshot.get("language_goal") or ""
        self.difficulty = snapshot.get("difficulty")
        self.seed = snapshot.get("seed")
        self.demonstration_frames = list(snapshot.get("demonstration_frames") or [])
        self.base_frames = list(snapshot.get("base_frames") or [])
        self.wrist_frames = list(snapshot.get("wrist_frames") or [])
        self.available_options = list(snapshot.get("available_options") or [])
        self.raw_solve_options = list(snapshot.get("raw_solve_options") or [])
        self.seg_vis = snapshot.get("seg_vis")
        self.is_demonstration = bool(snapshot.get("is_demonstration", False))
        self.non_demonstration_task_length = snapshot.get("non_demonstration_task_length")
        if reset_execution_frames:
            self.last_execution_frames = []
        else:
            self.last_execution_frames = list(snapshot.get("last_execution_frames") or [])

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

    def _software_call(self, method, *args, reset_execution_frames=False, **kwargs):
        assert self._software_session is not None
        payload = self._software_session.call(method, *args, **kwargs)
        self._apply_snapshot(payload.get("snapshot"), reset_execution_frames=reset_execution_frames)
        return payload.get("result")

    def load_episode(self, env_id, episode_idx):
        if self._backend_mode == "software_render_subprocess":
            try:
                return self._software_call(
                    "load_episode",
                    env_id,
                    episode_idx,
                    reset_execution_frames=True,
                )
            except SoftwareRenderUnsupportedError as exc:
                self._apply_snapshot(_build_default_snapshot(), reset_execution_frames=True)
                return None, _format_software_render_error(exc)
            except RemoteSessionError as exc:
                self._apply_snapshot(_build_default_snapshot(), reset_execution_frames=True)
                return None, f"Error initializing episode: {exc}"

        result = load_episode_gpu(self._session, env_id, episode_idx)
        self._sync_state(reset_execution_frames=True)
        return result

    def execute_action(self, action_idx, click_coords):
        if self._backend_mode == "software_render_subprocess":
            try:
                return self._software_call("execute_action", action_idx, click_coords)
            except SoftwareRenderUnsupportedError as exc:
                raise RuntimeError(_format_software_render_error(exc)) from exc
            except RemoteSessionError as exc:
                if exc.error_type == "ScrewPlanFailure":
                    raise ScrewPlanFailureError(f"screw plan failed: {exc}") from exc
                raise RuntimeError(str(exc)) from exc

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
        if self._backend_mode == "software_render_subprocess":
            return _snapshot_to_pil_image(self._snapshot, use_segmented=use_segmented)
        return self._session.get_pil_image(use_segmented=use_segmented)

    def update_observation(self, use_segmentation=True):
        if self._backend_mode == "software_render_subprocess":
            try:
                return self._software_call(
                    "update_observation",
                    use_segmentation,
                    reset_execution_frames=False,
                )
            except SoftwareRenderUnsupportedError as exc:
                return None, _format_software_render_error(exc)
            except RemoteSessionError as exc:
                return None, f"Error updating observation: {exc}"

        result = update_observation_gpu(self._session, use_segmentation)
        self._sync_state(reset_execution_frames=False)
        return result

    def get_reference_action(self):
        if self._backend_mode == "software_render_subprocess":
            try:
                return self._software_call("get_reference_action")
            except SoftwareRenderUnsupportedError as exc:
                return {
                    "ok": False,
                    "option_idx": None,
                    "option_label": "",
                    "option_action": "",
                    "need_coords": False,
                    "coords_xy": None,
                    "message": _format_software_render_error(exc),
                }
            except RemoteSessionError as exc:
                return {
                    "ok": False,
                    "option_idx": None,
                    "option_label": "",
                    "option_action": "",
                    "need_coords": False,
                    "coords_xy": None,
                    "message": str(exc),
                }

        result = self._session.get_reference_action()
        self._sync_state(reset_execution_frames=False)
        return result

    def close(self):
        if self._backend_mode == "software_render_subprocess":
            assert self._software_session is not None
            self._software_session.close()
            LOGGER.info(
                "ProcessSessionProxy closed software-render subprocess env=%s episode=%s",
                self.env_id,
                self.episode_idx,
            )
            return

        self._session.close()
        LOGGER.info("ProcessSessionProxy closed in-process env=%s episode=%s", self.env_id, self.episode_idx)
