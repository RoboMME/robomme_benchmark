"""Unmask 环境 after-drop 场景快照的辅助逻辑。

这个模块做的事情比较单一：
1. 针对少数指定的 unmask 环境，在固定 step 抓取一次场景快照；
2. 从环境对象里提取 cube / bin 的位置和配对关系；
3. 把这些信息序列化为 JSON，供后续回放、可视化或人工检查使用。

设计上尽量保持“旁路接入”：
- 不改环境内部实现；
- 只在运行时包一层 `env.step`；
- 达到目标 step 后写一次文件，之后不再重复写。
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

DEFAULT_CAPTURE_STEP: int = 33

# `env_id -> capture step` 的映射表。
# 目前这几个环境都在同一个时间点抓 after-drop 快照，但保留成字典，
# 是为了以后某个环境若需要不同的抓取时机，可以只改这里，不用改主流程。
#
# 这里不显式区分 swap / non-swap：
# 后面会在运行时查看 `base_env` 上是否存在 `cube_bin_pairs`，
# 再自动决定走哪条解析路径。
SNAPSHOT_ENVS: dict[str, int] = {
    "ButtonUnmask": DEFAULT_CAPTURE_STEP,
    "ButtonUnmaskSwap": DEFAULT_CAPTURE_STEP,
    "VideoUnmask": DEFAULT_CAPTURE_STEP,
    "VideoUnmaskSwap": DEFAULT_CAPTURE_STEP,
}


def _to_python_int(value) -> int:
    """把环境里常见的 Tensor / ndarray 计数值安全转成 Python int。

    环境内部的计数器有时是：
    - Python 原生 int
    - 0-d / 1-d 的 numpy 数组
    - 标量 torch.Tensor

    为了避免后续比较 step 时出现类型不一致，这里统一做一次“压平并取第一个元素”。
    对空 Tensor / 空数组 / None 统一回落为 0，保证调用方不需要额外判空。
    """
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        # 有些环境变量可能是空 Tensor，这里直接当作 0 处理。
        if value.numel() == 0:
            return 0
        # detach + cpu 是为了避免设备差异和梯度上下文污染。
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        # 统一拉平成一维，兼容 shape 不固定的情况。
        value = np.asarray(value).reshape(-1)[0].item()
    return int(value)


def _actor_position_xyz(actor) -> list[float]:
    """读取 actor 的世界坐标，并规范成长度为 3 的 Python 列表。

    返回值固定为 `[x, y, z]`，原因有两个：
    - JSON 序列化时，Python float/list 最稳定；
    - 下游可视化或检查脚本通常只需要位置，不需要完整 pose。

    如果 actor 缺失，就返回原点占位，避免上层因为某个对象不存在而直接失败。
    """
    if actor is None:
        return [0.0, 0.0, 0.0]

    # 不同 actor 实现可能提供 `pose` 属性，也可能只暴露 `get_pose()`。
    # 这里兼容两种风格，减少对具体环境实现的假设。
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    position = pose.p
    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()

    # 强制转成一维 numpy 数组，便于后面统一处理。
    position_np = np.asarray(position, dtype=np.float64).reshape(-1)
    if position_np.size < 3:
        # 极端情况下如果位置向量维度不足 3，这里补零，保证 JSON 结构恒定。
        padded = np.zeros(3, dtype=np.float64)
        padded[: position_np.size] = position_np
        position_np = padded
    return [float(position_np[0]), float(position_np[1]), float(position_np[2])]


def _snapshot_json_path(output_root: Path, env_id: str, episode: int, seed: int) -> Path:
    """拼出 after-drop 快照 JSON 的标准落盘路径。

    文件名里包含 env / episode / seed，便于：
    - 不同实验结果并存；
    - 人工排查时直接从文件名定位样本；
    - 后续脚本按固定命名规则批量读取。
    """
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
    """把单个 cube 组织成快照字典。

    这里只保留最核心的信息：
    - cube 自身名字；
    - cube 颜色；
    - cube 世界坐标；
    - 它理论上配对的 bin 编号和名字。

    这样既能用于直观检查，也能让下游脚本判断“当前物体是否放到了应有区域附近”。
    """
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
    """把所有 bin 组织成快照字典列表。

    `has_cube_under_bin` 表示“按环境配对逻辑，这个 bin 下方应当有一个 cube”。
    它不是视觉检测结果，而是根据环境元信息推导出来的语义标记。
    """
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
    """构造最终写入 JSON 的顶层 payload。"""
    return {
        "env_id": env_id,
        "episode": int(episode),
        "seed": int(seed),
        "difficulty": difficulty,
        # `inspect_this_timestep` 记录“理论上计划在哪个 step 抓图”，
        # 方便和真实抓取 step（`capture_elapsed_steps`）对照。
        "inspect_this_timestep": SNAPSHOT_ENVS[env_id],
        # 真实抓取时环境内部已经走到的 elapsed_steps。
        # 一般等于目标 step，但若外层逻辑跨过目标 step 才首次进入，也允许大于它。
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

    通过 `base_env.cube_bin_pairs` 是否存在自动区分 swap / non-swap。

    这里的核心难点不在“怎么读位置”，而在“怎么还原 cube 和 bin 的语义配对关系”：
    - swap 环境通常显式保存 `(cube, bin)` 配对；
    - non-swap 环境更多是按固定 index 隐式对应。

    所以这里分成两条路径，最后再汇总成统一 JSON 结构。
    """
    # `spawned_bins` / `color_names` 即使缺失，也退化成空列表，避免环境细节差异导致异常。
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])

    # 记录哪些 bin 在语义上“对应着一个目标 cube”。
    # 后面会写入 bin 快照中，帮助下游判断哪些 bin 是需要重点看的。
    bins_with_cubes: set[int] = set()
    cubes: list[dict] = []

    cube_bin_pairs = getattr(base_env, "cube_bin_pairs", None)
    if cube_bin_pairs:
        # ---- swap 路径：显式 (cube, bin) 配对 ----
        # 某些 swap 环境会额外提供 `bin_to_color`，可以更准确地恢复“这个 bin 对应什么颜色”。
        bin_to_color = dict(getattr(base_env, "bin_to_color", {}) or {})
        # 由于 actor 本身未必可 hash，也不一定适合作为字典键，
        # 这里用 `id(actor)` 建立“对象身份 -> bin 下标”的映射。
        bin_index_by_id = {id(b): i for i, b in enumerate(spawned_bins)}

        for pair_idx, pair in enumerate(cube_bin_pairs):
            # 容错：如果环境返回的 pair 结构不合法，就跳过，而不是中断整个快照。
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue

            cube_actor, bin_actor = pair
            paired_bin_index = bin_index_by_id.get(id(bin_actor))
            if paired_bin_index is not None:
                bins_with_cubes.add(paired_bin_index)

            cube_color = None
            if paired_bin_index is not None:
                # 优先采用环境显式给出的 bin->color 映射。
                cube_color = bin_to_color.get(paired_bin_index)
            if cube_color is None and pair_idx < len(color_names):
                # 如果缺少显式映射，则退回到配对顺序对应的颜色名。
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
        # 当前任务约定最多有 3 组目标 cube/bin，这里同时受 spawned_bins 长度约束。
        pair_count = min(3, len(spawned_bins))
        for pair_idx in range(pair_count):
            # 优先尝试 `target_cube_0/1/2` 这种按下标命名的属性。
            cube_actor = getattr(base_env, f"target_cube_{pair_idx}", None)
            if cube_actor is None and pair_idx < len(color_names):
                # 兼容另一种命名风格：`target_cube_red` / `target_cube_blue` 等。
                cube_actor = getattr(
                    base_env, f"target_cube_{color_names[pair_idx]}", None
                )
            if cube_actor is None:
                # 如果连目标 cube 都找不到，说明这组配对无法恢复，直接跳过。
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

    # cube 和 bin 分别构建，再统一拼顶层 payload，方便后续字段演进。
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
    """给支持的 unmask env 打补丁，在固定时机抓 after-drop 快照。

    这里采用最轻量的接入方式：保存原始 `env.step`，再在外面包一层。
    包装后的流程是：
    1. 先正常执行一步环境；
    2. 查看当前 `elapsed_steps` 是否达到抓取阈值；
    3. 若达到且还没写过快照，就收集信息并写入 JSON；
    4. 无论是否抓取，都把原始 step 的返回值原样交还上层。

    返回的 `state` 主要用于外层脚本查询“是否启用快照、文件写到哪里了”。
    """
    # 只有列在 `SNAPSHOT_ENVS` 中的环境才启用这套逻辑，
    # 这样不会影响其它任务环境。
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
    # 保留原始 step，避免在包装函数里递归调用自己。
    original_step = env.step

    def instrumented_step(action):
        # 先让环境正常推进，确保抓取的是“执行完当前动作后的场景”。
        step_result = original_step(action)
        if state["snapshot_written"]:
            # 快照只写一次；一旦成功落盘，后续 step 完全透传。
            return step_result

        # 用 `unwrapped` 访问底层环境，避免被 wrapper 层屏蔽内部字段。
        base_env = env.unwrapped
        elapsed_steps = _to_python_int(getattr(base_env, "elapsed_steps", 0))
        if elapsed_steps < capture_step:
            # 还没到抓取时机，直接返回。
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
        # 自动创建目录，避免调用方还要显式保证 `snapshots/` 已存在。
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # 更新状态，供外层脚本记录或日志打印使用。
        state["snapshot_written"] = True
        state["snapshot_json_path"] = path
        print(f"After-drop snapshot JSON: {path.resolve()}")
        return step_result

    # 直接替换实例上的 step 方法，实现运行时插桩。
    env.step = instrumented_step  # type: ignore[method-assign]
    return state
