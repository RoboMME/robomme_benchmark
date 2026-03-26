"""ButtonUnmaskInspect 的快照辅助逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


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

    # Robomme 中有的对象直接暴露 `pose`，有的通过 `get_pose()` 获取；
    # 这里兼容两种写法，避免快照代码依赖具体 actor 实现。
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    position = pose.p
    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()

    position_np = np.asarray(position, dtype=np.float64).reshape(-1)
    if position_np.size < 3:
        # 某些测试替身对象可能只给了部分坐标；补零后统一输出格式，
        # 下游 JSON 使用方就不需要处理长度不一致的情况。
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


def _button_unmask_swap_inspect_this_timestep() -> int:
    """返回 ButtonUnmaskSwap 需要抓取 after-drop 快照的时间步。"""
    return 33


def _collect_button_unmask_swap_snapshot(
    base_env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    capture_elapsed_steps: int,
) -> dict:
    """收集 ButtonUnmaskSwap 在指定时刻的关键场景信息。

    这里的目标不是序列化整个环境，而是提取排查这个任务最需要的结构化信息：
    - 每个 bin 的位置和索引。
    - 每个 cube 当前的位置、颜色，以及它理论上配对的是哪个 bin。
    - 哪些 bin 下方确实有 cube，便于判断 drop 后的遮挡/覆盖关系。
    """
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    cube_bin_pairs = list(getattr(base_env, "cube_bin_pairs", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])
    bin_to_color = dict(getattr(base_env, "bin_to_color", {}) or {})
    inspect_this_timestep = _button_unmask_swap_inspect_this_timestep()
    # 用对象 id 建一个反查表，后面可以从 `(cube, bin)` 配对直接拿到 bin 序号。
    bin_index_by_id = {id(bin_actor): idx for idx, bin_actor in enumerate(spawned_bins)}
    bins_with_cubes = set()
    cubes = []

    for pair_idx, pair in enumerate(cube_bin_pairs):
        # 数据集构造阶段如果留下了脏数据，这里直接跳过非法 pair，
        # 保证快照导出本身不要再因为格式问题崩掉。
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
            # 优先使用环境里“bin 索引 -> 颜色”的真值映射；
            # 如果拿不到，再退回到 pair 的顺序颜色，尽量补全调试信息。
            cube_color = color_names[pair_idx]

        cubes.append(
            {
                "name": getattr(cube_actor, "name", None),
                "color": cube_color,
                "position_xyz": _actor_position_xyz(cube_actor),
                "paired_bin_index": (
                    int(paired_bin_index) if paired_bin_index is not None else None
                ),
                "paired_bin_name": getattr(bin_actor, "name", None),
            }
        )

    bins = []
    for idx, bin_actor in enumerate(spawned_bins):
        # `has_cube_under_bin` 不是视觉检测结果，而是根据配对信息推导出来的，
        # 用来快速看哪些 bin 理应覆盖着 cube。
        bins.append(
            {
                "index": idx,
                "name": getattr(bin_actor, "name", None),
                "position_xyz": _actor_position_xyz(bin_actor),
                "has_cube_under_bin": idx in bins_with_cubes,
            }
        )

    return {
        "env_id": env_id,
        "episode": int(episode),
        "seed": int(seed),
        "difficulty": difficulty,
        "inspect_this_timestep": inspect_this_timestep,
        "capture_elapsed_steps": int(capture_elapsed_steps),
        "capture_phase": "after_drop",
        "cubes": cubes,
        "bins": bins,
    }


def install_snapshot_for_step(
    env: gym.Env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    output_dir: Path,
) -> dict:
    """给 `env.step` 打补丁，在合适时机抓取 after-drop 快照。

    返回一个可变状态字典，闭包会原地更新两个字段：
    - `snapshot_written`: 是否已经成功写过快照。
    - `snapshot_json_path`: 快照的实际路径。

    只有 `ButtonUnmaskSwap` 需要这段逻辑；其他环境直接返回默认状态。
    """
    state: dict = {"snapshot_written": False, "snapshot_json_path": None}
    if env_id != "ButtonUnmaskSwap":
        return state

    original_step = env.step

    def instrumented_step(action):
        # 先保持环境原有 step 行为，再根据 step 后的状态决定是否抓快照。
        step_result = original_step(action)
        if state["snapshot_written"]:
            # 每局只抓一次，避免后续 step 反复覆盖同一份 JSON。
            return step_result

        base_env = env.unwrapped
        elapsed_steps = _to_python_int(getattr(base_env, "elapsed_steps", 0))
        if elapsed_steps < _button_unmask_swap_inspect_this_timestep():
            # 还没走到关注的“drop 后”时间点，继续执行即可。
            return step_result

        snapshot_payload = _collect_button_unmask_swap_snapshot(
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
        # 闭包通过共享状态把结果带回外层，便于 episode 结束时给出明确提示。
        state["snapshot_written"] = True
        state["snapshot_json_path"] = path
        print(f"After-drop snapshot JSON: {path.resolve()}")
        return step_result

    env.step = instrumented_step  # type: ignore[method-assign]
    return state
