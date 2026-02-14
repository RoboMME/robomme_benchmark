"""Utilities for picking out pickup tasks for failure recovery."""
from typing import Any, Iterable, List, Tuple, Union

from .planner import solve_pickup, solve_pickup_bin
import torch

TaskEntry = Union[dict, tuple, list]

FAIL_GRASP_MODES = ("xy", "z")


def _get_demo_flag(task: TaskEntry) -> bool:
    """Return the demonstration flag, defaulting to False when missing."""
    if isinstance(task, dict):
        if "demonstration" in task:
            return bool(task.get("demonstration"))
        return bool(task.get("demo", False))
    if isinstance(task, (list, tuple)) and len(task) >= 3:
        return bool(task[2])
    return False


def _extract_solve(task: TaskEntry) -> Any:
    """Fetch the solve callable from a task entry if present."""
    if isinstance(task, dict):
        return task.get("solve")
    if isinstance(task, (list, tuple)) and len(task) >= 5:
        return task[4]
    return None


def _resolve_pickup_solver(solve_callable: Any):
    """Return the pickup solver callable referenced by a task, if any."""
    if solve_callable is None:
        return None
    if isinstance(solve_callable, (list, tuple)):
        for cb in solve_callable:
            solver = _resolve_pickup_solver(cb)
            if solver:
                return solver
        return None

    if solve_callable in (solve_pickup, solve_pickup_bin):
        return solve_callable

    name = getattr(solve_callable, "__name__", "")
    if name == "solve_pickup":
        return solve_pickup
    if name == "solve_pickup_bin":
        return solve_pickup_bin

    underlying = getattr(solve_callable, "func", None)
    if underlying and underlying is not solve_callable:
        solver = _resolve_pickup_solver(underlying)
        if solver:
            return solver

    code_obj = getattr(solve_callable, "__code__", None)
    if code_obj:
        if "solve_pickup_bin" in code_obj.co_names:
            return solve_pickup_bin
        if "solve_pickup" in code_obj.co_names:
            return solve_pickup

    wrapped = getattr(solve_callable, "__wrapped__", None)
    if wrapped and wrapped is not solve_callable:
        solver = _resolve_pickup_solver(wrapped)
        if solver:
            return solver

    return None


def _normalize_single_obj(obj: Any) -> Any:
    """
    Some tasks store a single segment as a list/tuple with one element.
    For pickup we only need the underlying object, not the container.
    """
    if isinstance(obj, (list, tuple)):
        return obj[0] if obj else None
    return obj


def _solve_refs_pickup(solve_callable: Any) -> bool:
    """
    Check whether a solve callable eventually calls `solve_pickup` or
    `solve_pickup_bin` without executing it. Handles plain callables,
    functools.partial, and containers.
    """
    return _resolve_pickup_solver(solve_callable) is not None


def task4recovery(task_list: Iterable[TaskEntry]) -> Tuple[List[int], List[TaskEntry]]:
    """
    Pass task_list, return indices and task entries where solve uses solve_pickup or solve_pickup_bin
    and demonstration=False.

    Args:
        task_list: Sequential task list (dict or old format tuple/list).

    Returns:
        (pickup_indices, pickup_tasks)
    """
    pickup_indices: List[int] = []
    pickup_tasks: List[TaskEntry] = []

    for idx, task in enumerate(task_list):
        if _get_demo_flag(task):
            continue
        solve_callable = _extract_solve(task)
        if _solve_refs_pickup(solve_callable):
            pickup_indices.append(idx)
            pickup_tasks.append(task)

    return pickup_indices, pickup_tasks


def _make_fail_grasp_solve(solve_callable: Any, obj: Any, mode: str):
    """Wrap a solve callable to force fail_grasp=True with a specific failure mode."""
    solver = _resolve_pickup_solver(solve_callable)
    target_obj = _normalize_single_obj(obj)

    def _wrapped(env, planner):
        # If we can directly call a pickup solver, force fail_grasp there to ensure failure injection.
        if solver is not None:
            try:
                return solver(env, planner, obj=target_obj, fail_grasp=True, mode=mode)
            except TypeError:
                return solver(env, planner, obj=target_obj, fail_grasp=True)

        if solve_callable is None:
            return solve_pickup(env, planner, obj=target_obj, fail_grasp=True, mode=mode)

        try:
            return solve_callable(env, planner, fail_grasp=True, mode=mode)
        except TypeError:
            # The callable does not accept fail_grasp or mode; run it without the extra keywords then fall back.
            try:
                return solve_callable(env, planner, fail_grasp=True)
            except TypeError:
                try:
                    return solve_callable(env, planner)
                except TypeError:
                    return solve_pickup(env, planner, obj=target_obj, fail_grasp=True, mode=mode)

    return _wrapped


def inject_fail_grasp(task_list: Iterable[TaskEntry], generator: torch.Generator = None, mode: str = None):
    """
    Randomly select a pickup task, replace its solve with a version where fail_grasp=True.

    Args:
        task_list: Task list
        generator: torch.Generator, for reproducible random selection

    Returns:
        Index of modified task; return None if no pickup task exists.
    """
    pickup_indices, _ = task4recovery(task_list)
    if not pickup_indices:
        return None

    torch_gen = generator if isinstance(generator, torch.Generator) else None
    if torch_gen is not None:
        choice = torch.randint(0, len(pickup_indices), (1,), generator=torch_gen).item()
    else:
        choice = torch.randint(0, len(pickup_indices), (1,)).item()

    target_idx = pickup_indices[choice]
    task = task_list[target_idx]
    normalized_mode = mode.lower() if isinstance(mode, str) else mode
    if normalized_mode is None:
        if torch_gen is not None:
            mode_choice = FAIL_GRASP_MODES[torch.randint(0, len(FAIL_GRASP_MODES), (1,), generator=torch_gen).item()]
        else:
            mode_choice = FAIL_GRASP_MODES[torch.randint(0, len(FAIL_GRASP_MODES), (1,)).item()]
    else:
        if normalized_mode not in FAIL_GRASP_MODES:
            raise ValueError(f"Unknown fail grasp mode {mode!r}")
        mode_choice = normalized_mode
    if isinstance(task, dict):
        obj = task.get("segment")
        solve_callable = task.get("solve")
        task["solve"] = _make_fail_grasp_solve(solve_callable, obj, mode_choice)
        task["fail_grasp_mode"] = mode_choice
        task["fail_grasp_injected"] = True
    return target_idx
