"""ZeroGPU compatibility helpers for RoboMME."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar


LOGGER = logging.getLogger("robomme.zerogpu")
F = TypeVar("F", bound=Callable[..., Any])


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


@gpu_task(duration=75)
def load_episode_gpu(session, env_id, episode_idx):
    return session.load_episode(env_id, episode_idx)


@gpu_task(duration=120)
def execute_action_gpu(session, action_idx, click_coords):
    return session.execute_action(action_idx, click_coords)


@gpu_task(duration=90)
def update_observation_gpu(session, use_segmentation=True):
    return session.update_observation(use_segmentation=use_segmentation)
