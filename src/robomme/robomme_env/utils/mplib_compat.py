"""Compatibility shim for mplib API differences between 0.1.x and 0.2.x.

The installed ManiSkill motion-planning code was written against mplib 0.1.x,
which expects numpy arrays for poses and accepts ``use_point_cloud`` in planning
calls.  mplib 0.2.x changes ``Planner.set_base_pose`` to expect an
``mplib.pymp.Pose`` object, removes ``use_point_cloud`` from ``plan_screw``,
and renames ``plan_qpos_to_pose`` to ``plan_pose``.

This module monkey-patches ``mplib.Planner`` at import time so legacy
callers work against either version.  It is a no-op on 0.1.x because the wrapped
functions still accept the legacy signatures.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, List, Tuple

import numpy as np


_GLOBAL_PATCH_INSTALLED = False

# Legacy RRT* kwargs that 0.1.x accepted but 0.2.x's ``plan_pose`` does not.
# ``rrt_range`` and ``planning_time`` are accepted by 0.2.x and are left to the
# signature-based filter; ``planner_name`` and ``use_point_cloud`` are always
# dropped on 0.2.x.
_DROPPED_RRT_KWARGS = {"planner_name", "use_point_cloud"}


def _mplib_pose_type():
    """Return the mplib pose type (module-level to defer import)."""
    try:
        from mplib.pymp import Pose as MPose

        return MPose
    except Exception:
        return None


def _to_mplib_pose(pose: Any) -> Any:
    """Convert a 7-DOF pose array or sapien.Pose to mplib.pymp.Pose.

    mplib 0.1.x accepted numpy arrays; 0.2.x requires ``mplib.pymp.Pose``.
    """
    MPose = _mplib_pose_type()
    if MPose is None:
        return pose

    if hasattr(pose, "p") and hasattr(pose, "q"):
        # sapien.Pose or similar pose object
        return MPose(pose.p, pose.q)

    arr = np.asarray(pose).reshape(-1)
    if arr.size == 7:
        return MPose(arr[:3], arr[3:])
    raise ValueError(f"Cannot convert pose of shape {arr.shape} to mplib pose")


def filter_kwargs_for_callable(fn: Callable, kwargs: dict[str, Any]) -> Tuple[dict[str, Any], List[str]]:
    """Return (supported_kwargs, dropped_kwargs) for a callable.

    If the callable accepts ``**kwargs``, return the original dict unchanged.
    """
    sig = inspect.signature(fn)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs, []
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(set(kwargs.keys()) - supported)
    return filtered, dropped


def _detect_version_family() -> str:
    """Return '0.2.x' or '0.1.x' based on Planner.__init__ signature."""
    import mplib

    sig = inspect.signature(mplib.Planner.__init__)
    params = list(sig.parameters.keys())
    if params[1:3] == ["urdf", "move_group"]:
        return "0.2.x"
    return "0.1.x"


def _call_set_base_pose_compat(planner, pose, _original=None):
    """Call planner.set_base_pose with the pose format the version expects."""
    if _detect_version_family() == "0.2.x":
        pose = _to_mplib_pose(pose)
    elif _mplib_pose_type() is not None and isinstance(pose, _mplib_pose_type()):
        # 0.1.x expects array, but caller gave mplib Pose
        pose = np.concatenate([pose.p, pose.q])

    try:
        if _original is not None:
            return _original(planner, pose)
        return getattr(planner, "set_base_pose")(pose)
    except Exception as e:
        warnings.warn(f"[mplib-compat] set_base_pose failed: {e!r}")
        raise


def _call_plan_screw_compat(planner, goal_pose, current_qpos, _original=None, **kwargs):
    """Call planner.plan_screw with kwargs filtered to the installed signature."""
    fn = _original if _original is not None else getattr(type(planner), "plan_screw")
    goal_pose = _to_mplib_pose(goal_pose)
    filtered, dropped = filter_kwargs_for_callable(fn, kwargs)
    if dropped:
        print(f"[mplib-compat] plan_screw dropped unsupported kwargs: {dropped}")
    if _original is not None:
        return _original(planner, goal_pose, current_qpos, **filtered)
    return getattr(planner, "plan_screw")(goal_pose, current_qpos, **filtered)


def _call_plan_pose_compat(
    planner,
    goal_pose,
    current_qpos,
    _original: Callable | None = None,
    _alias_name: str = "plan_pose",
    **kwargs,
):
    """Call ``plan_pose`` (0.2.x) or ``plan_qpos_to_pose`` (0.1.x).

    Converts the goal pose to ``mplib.pymp.Pose`` when required and drops
    legacy RRT* kwargs that no longer exist in 0.2.x.
    """
    fn = _original if _original is not None else getattr(type(planner), _alias_name)
    goal_pose = _to_mplib_pose(goal_pose)

    # Always drop kwargs that are definitely gone in 0.2.x; the signature-based
    # filter then keeps only what the concrete implementation accepts
    # (e.g. ``rrt_range`` and ``planning_time`` are kept for ``plan_pose``).
    kwargs = {k: v for k, v in kwargs.items() if k not in _DROPPED_RRT_KWARGS}
    filtered, dropped = filter_kwargs_for_callable(fn, kwargs)
    if dropped:
        print(f"[mplib-compat] {_alias_name} dropped unsupported kwargs: {dropped}")
    return fn(planner, goal_pose, current_qpos, **filtered)


def install_global_mplib_compat() -> dict[str, Any]:
    """Monkey-patch ``mplib.Planner`` class so legacy callers work on 0.2.x.

    Safe to call multiple times; no-op on 0.1.x because the wrapped functions
    still accept the legacy signatures.  Returns a dict indicating which
    methods were patched.
    """
    global _GLOBAL_PATCH_INSTALLED
    if _GLOBAL_PATCH_INSTALLED:
        return {"already_installed": True}

    import mplib

    patched = {}
    version_family = _detect_version_family()

    orig_set_base_pose = mplib.Planner.set_base_pose

    def wrapped_set_base_pose(self, pose):
        return _call_set_base_pose_compat(self, pose, _original=orig_set_base_pose)

    mplib.Planner.set_base_pose = wrapped_set_base_pose
    patched["set_base_pose"] = True

    orig_plan_screw = mplib.Planner.plan_screw

    def wrapped_plan_screw(self, goal_pose, current_qpos, **kwargs):
        return _call_plan_screw_compat(self, goal_pose, current_qpos, _original=orig_plan_screw, **kwargs)

    mplib.Planner.plan_screw = wrapped_plan_screw
    patched["plan_screw"] = True

    if version_family == "0.2.x" and hasattr(mplib.Planner, "plan_pose"):
        # 0.2.x renamed plan_qpos_to_pose -> plan_pose.  Wrap plan_pose so
        # legacy callers that pass numpy arrays or obsolete kwargs still work,
        # and add a plan_qpos_to_pose alias for code written against 0.1.x.
        orig_plan_pose = mplib.Planner.plan_pose

        def wrapped_plan_pose(self, goal_pose, current_qpos, **kwargs):
            return _call_plan_pose_compat(
                self, goal_pose, current_qpos, _original=orig_plan_pose, _alias_name="plan_pose", **kwargs
            )

        mplib.Planner.plan_pose = wrapped_plan_pose
        patched["plan_pose"] = True

        mplib.Planner.plan_qpos_to_pose = wrapped_plan_pose
        patched["plan_qpos_to_pose_alias"] = True
    elif hasattr(mplib.Planner, "plan_qpos_to_pose"):
        # 0.1.x: wrap the native method for kwargs/ pose conversion safety.
        orig_plan_qpos_to_pose = mplib.Planner.plan_qpos_to_pose

        def wrapped_plan_qpos_to_pose(self, goal_pose, current_qpos, **kwargs):
            return _call_plan_pose_compat(
                self,
                goal_pose,
                current_qpos,
                _original=orig_plan_qpos_to_pose,
                _alias_name="plan_qpos_to_pose",
                **kwargs,
            )

        mplib.Planner.plan_qpos_to_pose = wrapped_plan_qpos_to_pose
        patched["plan_qpos_to_pose"] = True

    _GLOBAL_PATCH_INSTALLED = True
    return patched
