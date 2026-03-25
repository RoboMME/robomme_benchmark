import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..logging_utils import logger


# 这个模块实现的是“episode 级对象日志”的轻量辅助逻辑。
# 它和普通 logger.debug/info 不同，不负责输出人读的文本日志，
# 而是把环境里的对象快照、swap 事件、collision 摘要整理成一条 JSONL record。
#
# 设计约束：
# 1. state 直接挂在 env 对象上，避免 wrapper / env 双方各自维护副本。
# 2. 只保存最小必要字段，便于回放排查和后处理统计。
# 3. 所有值在最终写盘前都转成 json-able 的纯 Python 结构，避免 tensor / ndarray 泄漏到 json.dumps。
EPISODE_OBJECT_LOG_FILENAME = "episode_object_logs.jsonl"
_EPISODE_OBJECT_LOG_STATE_ATTR = "_episode_object_log_state"


def _to_python_scalar(value: Any) -> Any:
    """把 numpy 标量还原成原生 Python 标量。

    json.dumps 对大部分 Python 原生类型都能直接处理，但对 numpy scalar
    兼容性较差，所以这里先把它们转成 int / float / bool 等原生类型。
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_jsonable(value: Any) -> Any:
    """递归把输入值转换成可 JSON 序列化的结构。

    支持的常见输入包括：
    - torch.Tensor
    - np.ndarray / np scalar
    - dict
    - list / tuple

    这个函数是整个模块的最后一道保险，确保写盘前的数据不再依赖
    torch / numpy 的运行时类型。
    """
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return _to_python_scalar(value)


def _empty_episode_object_log_state() -> dict[str, list[Any]]:
    """创建一个新的空 episode 日志 state。

    每个 episode 都会围绕这几个字段累积信息：
    - bin_list: reset 后容器对象的静态快照
    - cube_list: reset 后 cube 对象的静态快照
    - target_cube_list: reset 后目标 cube 的静态快照
    - swap_events: episode 运行期间实际发生的 swap 事件
    - collision_events: episode 结束前整理出来的碰撞摘要
    """
    return {
        "bin_list": [],
        "cube_list": [],
        "target_cube_list": [],
        "swap_events": [],
        "collision_events": [],
    }


def _get_episode_object_log_state(env: Any) -> dict[str, list[Any]]:
    """读取 env 上挂载的日志 state；如果不存在则懒初始化。

    这里用“按需创建”而不是假设调用方一定先初始化，
    是为了让 helper 在测试桩、简化 env、异常恢复路径里也更稳健。
    """
    state = getattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, None)
    if not isinstance(state, dict):
        state = _empty_episode_object_log_state()
        setattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, state)
    return state


def init_episode_object_log_state(env: Any) -> None:
    """显式重置 env 上的 episode 对象日志 state。"""
    setattr(env, _EPISODE_OBJECT_LOG_STATE_ATTR, _empty_episode_object_log_state())


def extract_actor_world_position(actor: Any) -> Optional[list[float]]:
    """从 actor.pose.p 提取世界坐标，并统一成 `[x, y, z]`。

    返回 `None` 的情况：
    - actor 为空
    - actor 没有 pose
    - pose 上没有 p
    - 坐标长度不足 3

    这里不抛异常，而是尽量降级为 `None`，因为对象日志是旁路诊断信息，
    不应该因为单个 actor 取位姿失败而影响主流程。
    """
    if actor is None:
        return None

    pose = getattr(actor, "pose", None)
    if pose is None:
        return None

    position = getattr(pose, "p", None)
    if position is None:
        return None

    if isinstance(position, torch.Tensor):
        position = position.detach().cpu().numpy()
    else:
        position = np.asarray(position)

    position = position.reshape(-1)
    if position.size < 3:
        return None
    return [float(position[0]), float(position[1]), float(position[2])]


def _serialize_actor_item(item: Any) -> Optional[dict[str, Any]]:
    """把单个 `{actor, color, ...}` 项压缩成最小可持久化结构。

    当前只保留三类信息：
    - name: 后续定位对象最稳定的标识
    - position: reset 时或记录当下的世界坐标
    - color: 任务层面可读的语义标签

    其他运行时细节故意不写入，以保持 schema 简单稳定。
    """
    if not isinstance(item, dict):
        return None
    actor = item.get("actor")
    return {
        "name": getattr(actor, "name", None),
        "position": extract_actor_world_position(actor),
        "color": item.get("color"),
    }


def _serialize_actor_list(items: Any) -> list[dict[str, Any]]:
    """批量序列化 actor 列表，并自动跳过非法项。"""
    if not isinstance(items, list):
        return []
    serialized = []
    for item in items:
        payload = _serialize_actor_item(item)
        if payload is not None:
            serialized.append(payload)
    return serialized


def record_reset_objects(
    env: Any,
    *,
    bin_list: list[dict[str, Any]],
    cube_list: list[dict[str, Any]],
    target_cube_list: list[dict[str, Any]],
) -> None:
    """记录 episode reset 后的对象布局快照。

    这个函数通常在环境 `_initialize_episode()` 里调用，语义上表示：
    “本 episode 的初始对象配置已经确定，可以把之后做分析需要的最小快照保存下来。”

    注意这里会清空 `swap_events`：
    reset 意味着新的 episode 开始，之前 episode 的 swap 记录不能残留。
    `collision_events` 不在这里清空，是因为通常会通过重新初始化整个 state
    或者只在 close 前追加 collision summary；这里保持最小改动面。
    """
    state = _get_episode_object_log_state(env)
    state["bin_list"] = _serialize_actor_list(bin_list)
    state["cube_list"] = _serialize_actor_list(cube_list)
    state["target_cube_list"] = _serialize_actor_list(target_cube_list)
    state["swap_events"] = []


def append_episode_object_swap_event(
    env: Any,
    *,
    swap_index: int,
    object_a: Any,
    object_b: Any,
) -> None:
    """追加一条 swap 事件。

    这里记录的是“实际 resolve 出来的交换对象”，不是配置里的抽象 slot。
    因此 object_a / object_b 只存 actor.name，避免把完整 actor 结构带进日志。
    """
    state = _get_episode_object_log_state(env)
    state["swap_events"].append(
        {
            "swap_index": int(swap_index),
            "object_a": getattr(object_a, "name", None),
            "object_b": getattr(object_b, "name", None),
        }
    )


def build_episode_object_collision_event(
    contact_summary: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """把 contact summary 规整成稳定的 collision event schema。

    输入通常来自 `get_swap_contact_summary(...)`。
    即使上游给的是不完整 dict，这里也会补默认值，保证最终 schema 稳定。
    """
    summary = contact_summary or {}
    return {
        "swap_contact_detected": bool(summary.get("swap_contact_detected", False)),
        "first_contact_step": summary.get("first_contact_step"),
        "contact_pairs": _to_jsonable(summary.get("contact_pairs", [])),
        "max_force_norm": float(summary.get("max_force_norm", 0.0)),
        "max_force_pair": summary.get("max_force_pair"),
        "max_force_step": summary.get("max_force_step"),
        "pair_max_force": {
            str(key): float(value)
            for key, value in (summary.get("pair_max_force", {}) or {}).items()
        },
    }


def append_episode_object_collision_event(
    env: Any,
    *,
    contact_summary: Optional[dict[str, Any]],
) -> None:
    """把 episode 级碰撞摘要追加到 state 中。

    当前调用方式通常是：
    - step 内持续累计碰撞统计
    - close 前把累计结果压缩成一条 summary
    - 在这里 append 进 `collision_events`

    也就是说，这里的 event 更接近“summary event”，不是逐 step 碰撞流。
    """
    state = _get_episode_object_log_state(env)
    state["collision_events"].append(
        build_episode_object_collision_event(contact_summary)
    )


def build_episode_object_log_record(
    env: Any,
    *,
    env_id: Optional[str],
    episode: Optional[int],
    seed: Optional[int],
) -> dict[str, Any]:
    """组装最终落盘的一整条 episode record。

    这里会把 env 上累积好的 state 和 wrapper 侧上下文拼起来。
    输出 schema 对应 JSONL 里的一行。
    """
    state = _get_episode_object_log_state(env)
    return {
        "env": env_id,
        "episode": None if episode is None else int(episode),
        "seed": None if seed is None else int(seed),
        "bin_list": _to_jsonable(state.get("bin_list", [])),
        "cube_list": _to_jsonable(state.get("cube_list", [])),
        "target_cube_list": _to_jsonable(state.get("target_cube_list", [])),
        "swap_events": _to_jsonable(state.get("swap_events", [])),
        "collision_events": _to_jsonable(state.get("collision_events", [])),
    }


def append_episode_object_log_record(
    output_root: Any,
    record: dict[str, Any],
) -> Optional[Path]:
    """把一条 episode record 追加写入 `<output_root>/episode_object_logs.jsonl`。

    采用 JSONL 的原因：
    - append-only，适合逐 episode 持续写入
    - 后处理简单，逐行读取即可
    - 单条 record 写入失败时，影响面比整体 JSON 文件更小

    写入失败不会抛出异常中断主流程，而是记录 debug 信息并返回 `None`。
    对象日志属于辅助诊断产物，不应该因为落盘失败破坏环境回放或数据录制。
    """
    jsonl_path = Path(output_root) / EPISODE_OBJECT_LOG_FILENAME
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.debug(f"Warning: failed to append episode object log to {jsonl_path}: {exc}")
        return None
    return jsonl_path
