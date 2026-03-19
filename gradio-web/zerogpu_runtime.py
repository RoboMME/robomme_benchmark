"""ZeroGPU compatibility helpers for RoboMME."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, TypeVar


LOGGER = logging.getLogger("robomme.zerogpu")
F = TypeVar("F", bound=Callable[..., Any])
_WORKER_RUNTIME_LOGGED = False


def _noop_gpu_decorator(fn: F) -> F:
    return fn


def _resolve_spaces_gpu():
    try:
        import spaces  # type: ignore
    except ImportError:
        LOGGER.info("spaces package unavailable; using no-op GPU decorator")
        return None
    return getattr(spaces, "GPU", None)


def gpu_task(*, duration: int, size: str = "large") -> Callable[[F], F]:
    """Wrap a callable with spaces.GPU when available, otherwise no-op locally."""
    gpu_decorator = _resolve_spaces_gpu()
    if gpu_decorator is None:
        return _noop_gpu_decorator

    def _decorate(fn: F) -> F:
        wrapped = gpu_decorator(duration=duration, size=size)(fn)
        return functools.wraps(fn)(wrapped)

    return _decorate


def _prepare_zerogpu_worker_runtime() -> None:
    """Log the effective GPU worker runtime once per process for diagnostics."""
    global _WORKER_RUNTIME_LOGGED
    if _WORKER_RUNTIME_LOGGED:
        return

    snapshot = {
        "SPACES_ZERO_GPU": os.getenv("SPACES_ZERO_GPU"),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES"),
        "NVIDIA_DRIVER_CAPABILITIES": os.getenv("NVIDIA_DRIVER_CAPABILITIES"),
        "VK_ICD_FILENAMES": os.getenv("VK_ICD_FILENAMES"),
        "__EGL_VENDOR_LIBRARY_FILENAMES": os.getenv("__EGL_VENDOR_LIBRARY_FILENAMES"),
        "ROBOMME_RENDER_BACKEND": os.getenv("ROBOMME_RENDER_BACKEND"),
    }
    LOGGER.info("ZeroGPU worker runtime snapshot: %s", snapshot)
    LOGGER.debug(
        "Deferring SAPIEN render-device probing until the render backend runtime is configured"
    )

    _WORKER_RUNTIME_LOGGED = True


@gpu_task(duration=75)
def load_episode_gpu(session, env_id, episode_idx):
    _prepare_zerogpu_worker_runtime()
    return session.load_episode(env_id, episode_idx)


@gpu_task(duration=120)
def execute_action_gpu(session, action_idx, click_coords):
    _prepare_zerogpu_worker_runtime()
    return session.execute_action(action_idx, click_coords)


@gpu_task(duration=90)
def update_observation_gpu(session, use_segmentation=True):
    _prepare_zerogpu_worker_runtime()
    return session.update_observation(use_segmentation=use_segmentation)
