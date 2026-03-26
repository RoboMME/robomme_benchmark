"""Unmask env 的 after-drop 快照辅助逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

DEFAULT_CAPTURE_STEP: int = 33

# env_id → capture step。新增 env 只需加一行；
# swap / non-swap 由运行时 base_env 属性自动检测。
SNAPSHOT_ENVS: dict[str, int] = {
    "ButtonUnmask": DEFAULT_CAPTURE_STEP,
    "ButtonUnmaskSwap": DEFAULT_CAPTURE_STEP,
    "VideoUnmask": DEFAULT_CAPTURE_STEP,
    "VideoUnmaskSwap": DEFAULT_CAPTURE_STEP,
}


def _to_python_int(value) -> int:
    """把环境里常见的 Tensor / ndarray 计数值安全转成 Python int。"""
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        value = np.asarray(value).reshape(-1)[0].item()
    return int(value)


def _actor_position_xyz(actor) -> list[float]:
    """读取 actor 的世界坐标，并规范成长度为 3 的 Python 列表。"""
    if actor is None:
        return [0.0, 0.0, 0.0]

    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    position = pose.p
    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()

    position_np = np.asarray(position, dtype=np.float64).reshape(-1)
    if position_np.size < 3:
        padded = np.zeros(3, dtype=np.float64)
        padded[: position_np.size] = position_np
        position_np = padded
    return [float(position_np[0]), float(position_np[1]), float(position_np[2])]


def _snapshot_json_path(output_root: Path, env_id: str, episode: int, seed: int) -> Path:
    """拼出 after-drop 快照 JSON 的标准落盘路径。"""
    return (
        output_root
        / "snapshots"
        / f"{env_id}_ep{episode}_seed{seed}_after_drop.json"
    )


def _build_cube_snapshot(
    cube_actor,
    cube_color: str | None,
    paired_bin_index: int | None,
    paired_bin_actor,
) -> dict:
    return {
        "name": getattr(cube_actor, "name", None),
        "color": cube_color,
        "position_xyz": _actor_position_xyz(cube_actor),
        "paired_bin_index": (
            int(paired_bin_index) if paired_bin_index is not None else None
        ),
        "paired_bin_name": getattr(paired_bin_actor, "name", None),
    }


def _build_bins_snapshot(spawned_bins: list, bins_with_cubes: set[int]) -> list[dict]:
    bins = []
    for idx, bin_actor in enumerate(spawned_bins):
        bins.append(
            {
                "index": idx,
                "name": getattr(bin_actor, "name", None),
                "position_xyz": _actor_position_xyz(bin_actor),
                "has_cube_under_bin": idx in bins_with_cubes,
            }
        )
    return bins


def _build_snapshot_payload(
    *,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    capture_elapsed_steps: int,
    cubes: list[dict],
    bins: list[dict],
) -> dict:
    return {
        "env_id": env_id,
        "episode": int(episode),
        "seed": int(seed),
        "difficulty": difficulty,
        "inspect_this_timestep": SNAPSHOT_ENVS[env_id],
        "capture_elapsed_steps": int(capture_elapsed_steps),
        "capture_phase": "after_drop",
        "cubes": cubes,
        "bins": bins,
    }


def _collect_snapshot(
    base_env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    capture_elapsed_steps: int,
) -> dict:
    """收集 unmask 环境在固定 after-drop 时刻的场景信息。

    通过 base_env.cube_bin_pairs 是否存在自动区分 swap / non-swap。
    """
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])

    bins_with_cubes: set[int] = set()
    cubes: list[dict] = []

    cube_bin_pairs = getattr(base_env, "cube_bin_pairs", None)
    if cube_bin_pairs:
        # ---- swap 路径：显式 (cube, bin) 配对 ----
        bin_to_color = dict(getattr(base_env, "bin_to_color", {}) or {})
        bin_index_by_id = {id(b): i for i, b in enumerate(spawned_bins)}

        for pair_idx, pair in enumerate(cube_bin_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue

            cube_actor, bin_actor = pair
            paired_bin_index = bin_index_by_id.get(id(bin_actor))
            if paired_bin_index is not None:
                bins_with_cubes.add(paired_bin_index)

            cube_color = None
            if paired_bin_index is not None:
                cube_color = bin_to_color.get(paired_bin_index)
            if cube_color is None and pair_idx < len(color_names):
                cube_color = color_names[pair_idx]

            cubes.append(
                _build_cube_snapshot(
                    cube_actor=cube_actor,
                    cube_color=cube_color,
                    paired_bin_index=paired_bin_index,
                    paired_bin_actor=bin_actor,
                )
            )
    else:
        # ---- non-swap 路径：按 index 隐式配对 ----
        pair_count = min(3, len(spawned_bins))
        for pair_idx in range(pair_count):
            cube_actor = getattr(base_env, f"target_cube_{pair_idx}", None)
            if cube_actor is None and pair_idx < len(color_names):
                cube_actor = getattr(
                    base_env, f"target_cube_{color_names[pair_idx]}", None
                )
            if cube_actor is None:
                continue

            bin_actor = spawned_bins[pair_idx]
            bins_with_cubes.add(pair_idx)
            cube_color = color_names[pair_idx] if pair_idx < len(color_names) else None
            cubes.append(
                _build_cube_snapshot(
                    cube_actor=cube_actor,
                    cube_color=cube_color,
                    paired_bin_index=pair_idx,
                    paired_bin_actor=bin_actor,
                )
            )

    bins = _build_bins_snapshot(spawned_bins, bins_with_cubes)
    return _build_snapshot_payload(
        env_id=env_id,
        episode=episode,
        seed=seed,
        difficulty=difficulty,
        capture_elapsed_steps=capture_elapsed_steps,
        cubes=cubes,
        bins=bins,
    )


def install_snapshot_for_step(
    env: gym.Env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> dict:
    """给支持的 unmask env 打补丁，在固定时机抓 after-drop 快照。"""
    snapshot_enabled = env_id in SNAPSHOT_ENVS
    state: dict = {
        "snapshot_enabled": snapshot_enabled,
        "snapshot_written": False,
        "snapshot_json_path": None,
        "capture_phase": "after_drop" if snapshot_enabled else None,
        "expected_capture_step": (
            SNAPSHOT_ENVS[env_id] if snapshot_enabled else None
        ),
    }
    if not snapshot_enabled:
        return state

    capture_step = SNAPSHOT_ENVS[env_id]
    original_step = env.step

    def instrumented_step(action):
        step_result = original_step(action)
        if state["snapshot_written"]:
            return step_result

        base_env = env.unwrapped
        elapsed_steps = _to_python_int(getattr(base_env, "elapsed_steps", 0))
        if elapsed_steps < capture_step:
            return step_result

        snapshot_payload = _collect_snapshot(
            base_env=base_env,
            env_id=env_id,
            episode=episode,
            seed=seed,
            difficulty=difficulty,
            capture_elapsed_steps=elapsed_steps,
        )
        path = _snapshot_json_path(
            output_root=output_dir, env_id=env_id, episode=episode, seed=seed
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state["snapshot_written"] = True
        state["snapshot_json_path"] = path
        print(f"After-drop snapshot JSON: {path.resolve()}")
        return step_result

    env.step = instrumented_step  # type: ignore[method-assign]
    return state
