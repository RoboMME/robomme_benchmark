"""Unmask 环境 after-drop 场景快照与整局 bin-bin collision 监控逻辑。

这个模块做的事情比较单一：
1. 针对少数指定的 unmask 环境，在固定 step 抓取一次场景快照；
2. 从环境对象里提取 cube / bin 的位置和配对关系；
3. 从 episode 开始持续监控到结束，记录是否出现过有效 bin-bin collision；
4. 把这些信息序列化为 JSON，供后续回放、可视化或人工检查使用。

设计上尽量保持“旁路接入”：
- 不改环境内部实现；
- 只在运行时包一层 `env.step`；
- 达到目标 step 后先写出 after-drop 场景快照；若后续首次检测到 collision，
  再回写同一路径 JSON 中的 `collision` 字段。
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

DEFAULT_CAPTURE_STEP: int = 33
DEFAULT_COLLISION_FORCE_EPS: float = 1e-6
DEFAULT_WORKSPACE_ABS_LIMIT: float = 5.0

# `env_id -> capture step` 的映射表。
# 目前这几个环境都在同一个时间点抓 after-drop 快照，但保留成字典，
# 是为了以后某个环境若需要不同的抓取时机，可以只改这里，不用改主流程。
SNAPSHOT_ENVS: dict[str, int] = {
    "ButtonUnmask": DEFAULT_CAPTURE_STEP,
    "ButtonUnmaskSwap": DEFAULT_CAPTURE_STEP,
    "VideoUnmask": DEFAULT_CAPTURE_STEP,
    "VideoUnmaskSwap": DEFAULT_CAPTURE_STEP,
}
NON_SWAP_SNAPSHOT_ENVS = {"ButtonUnmask", "VideoUnmask"}
SWAP_SNAPSHOT_ENVS = {"ButtonUnmaskSwap", "VideoUnmaskSwap"}
BUTTON_SNAPSHOT_ENVS = {"ButtonUnmask", "ButtonUnmaskSwap"}


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


def _force_norm(force_tensor) -> float | None:
    """把 contact force 统一规整成一个标量范数。

    `scene.get_pairwise_contact_forces(...)` 的返回类型在不同后端/封装下
    可能并不完全一致：
    - 常见情况是 shape 近似为 `(N, 3)` 的 Tensor / ndarray；
    - 也可能已经是某种可直接转 numpy 的对象；
    - 没有接触时，可能返回空数组 / 空 Tensor / None。

    这里的目标不是保留完整接触明细，而是回答一个更简单的问题：
    “这一对 actor 当前是否发生了足够明显的接触？”
    因此只取第一个三维力向量并计算 L2 范数，供上层和阈值比较。
    """
    if force_tensor is None:
        return None
    if isinstance(force_tensor, torch.Tensor):
        if force_tensor.numel() == 0:
            return None
        force_vector = force_tensor.reshape(-1, 3)[0].detach().cpu().to(torch.float64)
        return float(torch.linalg.vector_norm(force_vector).item())

    force_array = np.asarray(force_tensor, dtype=np.float64)
    if force_array.size == 0:
        return None
    return float(np.linalg.norm(force_array.reshape(-1, 3)[0]))


def _is_actor_in_workspace(
    actor,
    *,
    position_abs_limit: float = DEFAULT_WORKSPACE_ABS_LIMIT,
) -> bool:
    """判断 actor 是否仍处于工作区附近，而不是被临时挪到场外占位点。

    一些 Robomme 状态变化辅助函数会把对象直接 teleport 到 `[10, 10, 10]`
    一类的远离工作台位置，用作“暂时移出场景”的占位手段。此时这些对象之间
    即使发生接触，也不应被当作 after-drop 场景里的有效 bin-bin collision。

    这里用一个比较宽松的绝对坐标上界做过滤：
    - 正常工作区内的 bin 通常都在原点附近；
    - 被移出场景的占位坐标通常远大于这个阈值。
    """
    position_xyz = _actor_position_xyz(actor)
    return all(abs(coord) <= float(position_abs_limit) for coord in position_xyz)


def _step_has_bin_collision(
    base_env,
    *,
    force_eps: float = DEFAULT_COLLISION_FORCE_EPS,
) -> bool:
    """检测当前 step 是否存在任意 spawned bin 之间的接触。

    这里专门看 bin-bin 碰撞，而不是 cube-bin / cube-table 等其它接触，
    因为我们关心的是“episode 推进过程中，多个 bin 是否曾互相顶住”。

    判定策略保持尽量宽松且稳健：
    1. 从 `base_env.spawned_bins` 取出当前真正生成出来的 bin；
    2. 两两枚举 bin 组合，向物理场景查询 pairwise contact force；
    3. 只要任意一对的力范数超过一个很小的 epsilon，就认为当前 step 出现过碰撞。

    这里返回的是“这一帧是否检测到碰撞”。
    外层 `install_snapshot_for_step` 会把它累计成 episode 级状态：
    一旦某步检测到碰撞，最终 JSON 中的 `collision=True`，即表示
    “本 episode 任意时刻至少出现过一次有效 bin-bin 接触”。
    """
    scene = getattr(base_env, "scene", None)
    spawned_bins = [
        actor
        for actor in list(getattr(base_env, "spawned_bins", []) or [])
        if _is_actor_in_workspace(actor)
    ]
    if scene is None or len(spawned_bins) < 2:
        # 没有 scene 无法查询接触；bin 少于 2 个时也不存在 bin-bin 碰撞。
        return False

    for actor_a, actor_b in combinations(spawned_bins, 2):
        try:
            pair_force = scene.get_pairwise_contact_forces(actor_a, actor_b)
        except Exception:
            # 某些后端/对象组合可能不支持该查询；这里按“未知即无碰撞”跳过，
            # 避免单个 pair 的异常中断整次 rollout。
            continue

        force_norm = _force_norm(pair_force)
        if force_norm is not None and force_norm > float(force_eps):
            # 这里不继续统计碰撞对数量，因为最终 JSON 只需要一个布尔结果。
            return True
    return False


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


def _build_button_snapshot(button_actor, *, fallback_name: str | None = None) -> dict:
    """把单个 button 组织成快照字典。"""
    return {
        "name": getattr(button_actor, "name", None) or fallback_name,
        "position_xyz": _actor_position_xyz(button_actor),
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
    collision: bool,
    cubes: list[dict],
    bins: list[dict],
    buttons: list[dict] | None = None,
) -> dict:
    """构造最终写入 JSON 的顶层 payload。

    注意：
    - 空间相关字段描述的是 after-drop 抓图时刻；
    - `collision` 描述的是整局 episode 级结果。
    """
    payload = {
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
        "collision": bool(collision),
        "cubes": cubes,
        "bins": bins,
    }
    if buttons is not None:
        payload["buttons"] = buttons
    return payload


def _collect_swap_cubes_and_bins(base_env) -> tuple[list[dict], list[dict]]:
    """按 swap 环境的显式 `(cube, bin)` 配对收集 cube/bin 快照。"""
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])
    bin_to_color = dict(getattr(base_env, "bin_to_color", {}) or {})
    bin_index_by_id = {id(bin_actor): idx for idx, bin_actor in enumerate(spawned_bins)}

    bins_with_cubes: set[int] = set()
    cubes: list[dict] = []

    for pair_idx, pair in enumerate(list(getattr(base_env, "cube_bin_pairs", []) or [])):
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

    return cubes, _build_bins_snapshot(spawned_bins, bins_with_cubes)


def _collect_non_swap_cubes_and_bins(base_env) -> tuple[list[dict], list[dict]]:
    """按 non-swap 环境的固定 index 约定收集 cube/bin 快照。"""
    spawned_bins = list(getattr(base_env, "spawned_bins", []) or [])
    color_names = list(getattr(base_env, "color_names", []) or [])

    bins_with_cubes: set[int] = set()
    cubes: list[dict] = []

    pair_count = min(3, len(spawned_bins))
    for pair_idx in range(pair_count):
        cube_actor = getattr(base_env, f"target_cube_{pair_idx}", None)
        if cube_actor is None and pair_idx < len(color_names):
            cube_actor = getattr(base_env, f"target_cube_{color_names[pair_idx]}", None)
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

    return cubes, _build_bins_snapshot(spawned_bins, bins_with_cubes)


def _collect_buttons_snapshot(base_env, env_id: str) -> list[dict] | None:
    """按 env_id 采集 button 位置；非 button env 返回 `None`。"""
    if env_id not in BUTTON_SNAPSHOT_ENVS:
        return None

    buttons: list[dict] = []
    if env_id == "ButtonUnmask":
        button_actor = getattr(base_env, "button_left", None) or getattr(
            base_env, "button", None
        )
        if button_actor is not None:
            buttons.append(
                _build_button_snapshot(button_actor, fallback_name="button_left")
            )
        return buttons

    for attr_name in ("button_left", "button_right"):
        button_actor = getattr(base_env, attr_name, None)
        if button_actor is None:
            continue
        buttons.append(
            _build_button_snapshot(button_actor, fallback_name=attr_name)
        )
    return buttons


def _collect_snapshot(
    base_env,
    env_id: str,
    episode: int,
    seed: int,
    difficulty: str,
    capture_elapsed_steps: int,
    collision: bool,
) -> dict:
    """收集 unmask 环境在固定 after-drop 时刻的场景信息。

    分派规则只由 `env_id` 决定：
    - `ButtonUnmask` / `VideoUnmask` 走 non-swap 路径；
    - `ButtonUnmaskSwap` / `VideoUnmaskSwap` 走 swap 路径。
    """
    if env_id in NON_SWAP_SNAPSHOT_ENVS:
        cubes, bins = _collect_non_swap_cubes_and_bins(base_env)
    elif env_id in SWAP_SNAPSHOT_ENVS:
        cubes, bins = _collect_swap_cubes_and_bins(base_env)
    else:
        raise ValueError(f"Unsupported snapshot env_id: {env_id}")

    return _build_snapshot_payload(
        env_id=env_id,
        episode=episode,
        seed=seed,
        difficulty=difficulty,
        capture_elapsed_steps=capture_elapsed_steps,
        collision=collision,
        cubes=cubes,
        bins=bins,
        buttons=_collect_buttons_snapshot(base_env, env_id),
    )


def _write_snapshot_payload(path: Path, payload: dict) -> None:
    """把快照 payload 写到标准 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _rewrite_snapshot_collision(path: Path, collision: bool) -> bool:
    """仅回写已落盘 JSON 中的 `collision` 字段。

    返回值表示文件内容是否真的发生了变化，便于调用方避免重复日志。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_collision = bool(collision)
    if bool(payload.get("collision")) == target_collision:
        return False
    payload["collision"] = target_collision
    _write_snapshot_payload(path, payload)
    return True


def _update_collision_sticky_flag(state: dict, base_env) -> bool:
    """在当前环境状态上执行一次碰撞采样，并更新 sticky flag。"""
    if state["collision_detected"]:
        return False
    if not _step_has_bin_collision(base_env):
        return False
    state["collision_detected"] = True
    return True


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
    1. 在调用原始 `env.step` 前先采样一次当前状态；
    2. 再正常执行一步环境；
    3. 在 `env.step` 返回后再采样一次最终状态，并累计成 episode 级 sticky flag；
    4. 若达到抓取阈值且还没写过快照，就立刻写出 after-drop JSON；
    5. 若快照已写出、但此后第一次观察到 collision，就回写同一路径 JSON 中的 `collision`；
    6. 无论是否抓取/回写，都把原始 step 的返回值原样交还上层。

    返回的 `state` 主要用于外层脚本查询“是否启用快照、文件写到哪里了”。
    注意：JSON 中的空间字段仍对应 after-drop 抓图时刻，但 `collision`
    是整局 episode 级布尔值。
    """
    # 只有列在 `SNAPSHOT_ENVS` 中的环境才启用这套逻辑，
    # 这样不会影响其它任务环境。
    snapshot_enabled = env_id in SNAPSHOT_ENVS
    state: dict = {
        "snapshot_enabled": snapshot_enabled,
        "snapshot_written": False,
        "snapshot_json_path": None,
        "capture_phase": "after_drop" if snapshot_enabled else None,
        "collision_detected": False,
        "snapshot_collision_synced": False,
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
        # 在 step 前后各采样一次，降低“只在 step 中间短暂接触而末态已分离”时
        # 因单次末态采样导致漏检的概率。
        base_env = env.unwrapped
        collision_just_detected = _update_collision_sticky_flag(state, base_env)

        # 再让环境正常推进，抓取/回写逻辑继续以“执行完当前动作后的场景”为准。
        step_result = original_step(action)

        # 用 `unwrapped` 访问底层环境，避免被 wrapper 层屏蔽内部字段。
        base_env = env.unwrapped
        collision_just_detected = (
            _update_collision_sticky_flag(state, base_env) or collision_just_detected
        )
        elapsed_steps = _to_python_int(getattr(base_env, "elapsed_steps", 0))

        if not state["snapshot_written"]:
            if elapsed_steps < capture_step:
                # 还没到抓取时机，直接返回。
                return step_result

            # 一旦达到抓取阈值，就立刻基于当前底层环境状态生成 JSON。
            # 这里故意不等 episode 结束：
            # after-drop 是一个时点快照，不是 episode summary。
            snapshot_payload = _collect_snapshot(
                base_env=base_env,
                env_id=env_id,
                episode=episode,
                seed=seed,
                difficulty=difficulty,
                capture_elapsed_steps=elapsed_steps,
                collision=state["collision_detected"],
            )
            path = _snapshot_json_path(
                output_root=output_dir, env_id=env_id, episode=episode, seed=seed
            )
            _write_snapshot_payload(path, snapshot_payload)
            # 更新状态，供外层脚本记录或日志打印使用。
            # 注意 `snapshot_json_path` 只有在真正写盘成功后才会填入。
            state["snapshot_written"] = True
            state["snapshot_json_path"] = path
            state["snapshot_collision_synced"] = bool(state["collision_detected"])
            print(f"After-drop snapshot JSON: {path.resolve()}")
            return step_result

        if (
            collision_just_detected
            and not state["snapshot_collision_synced"]
            and state["snapshot_json_path"] is not None
        ):
            updated = _rewrite_snapshot_collision(
                state["snapshot_json_path"],
                collision=True,
            )
            if updated:
                state["snapshot_collision_synced"] = True
                print(
                    "After-drop snapshot JSON collision flag updated: "
                    f"{state['snapshot_json_path'].resolve()}"
                )
        return step_result

    # 直接替换实例上的 step 方法，实现运行时插桩。
    # 这里是实例级修改，不影响同类 env 的其它实例。
    env.step = instrumented_step  # type: ignore[method-assign]
    return state
