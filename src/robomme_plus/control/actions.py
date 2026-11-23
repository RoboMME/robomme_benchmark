"""Shared action normalization for policy and privileged recording wrappers."""

import numpy as np


def normalize_joint_action(native, action):
    """Respect task-specific controllers, then handle Panda/stick dimensions."""
    normalizer = getattr(native, "normalize_policy_action", None)
    if callable(normalizer):
        return normalizer(action)
    env_id = getattr(native, "task_name", getattr(getattr(native, "spec", None), "id", "<unknown_env>"))
    size = 7 if env_id in ("PatternLock", "RouteStick") else 8
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"[{env_id}] action values must be finite")
    if values.size < size:
        raise ValueError(f"[{env_id}] action must have at least {size} elements, got {values.size}")
    return values[:size]
