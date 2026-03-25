# 当前 object log 机制说明

本文解释 `/runs/replay_videos/episode_object_logs.jsonl` 对应的当前 logging 机制。这里的 logging 不是普通文本 logger，而是一套 episode 级别的结构化 JSONL 输出。当前统一入口模块是 `src/robomme/robomme_env/utils/logging/object_log.py`。

## 0. 最小总结

`object_log.py` 提供的是一组 episode 级结构化日志 helper，统一按模块别名调用：
- env 构造时调用 `objectlog.init_episode_log()`，准备空 state
- env reset/init 时调用 `objectlog.record_object(event=..., payload=...)`，重置本轮 episode state 并记录初始对象快照
- env step 期间按需调用 `objectlog.record_swap()`，记录实际发生的 swap 对
- `RecordWrapper.close()` 里调用 `_episode_object_log_flush()`，从 `swap_contact_state` 取 summary 后，统一经 `objectlog.flush_episode_log()` 写入 `episode_object_logs.jsonl`

`swap_contact_monitoring.py` 提供的是一条独立的“swap 接触统计”链路：
- env 构造时调用 `swapContact.new_swap_contact_state()`，准备空的碰撞统计 state
- env reset 时调用 `swapContact.reset_swap_contact_state()`，清空上一个 episode 的累计统计
- env step 结束后调用 `swapContact.detect_swap_contacts()`，在 swap 时间窗内持续累计接触信息
- `RecordWrapper._episode_object_log_flush()` 里调用 `swapContact.get_swap_contact_summary()`，把累计 state 压成 summary，再决定是否补进 `collision_events`

`RecordWrapper.py` 在这条链路里的职责很单一：
- 不负责生成对象快照和 swap 事件
- 只负责在 episode 结束时把 env 上已经累积好的 state 收口，并把 close 阶段拿到的 collision summary 一并落盘

## 0.1 `swap_contact_monitoring.py` 用到了哪些接口

`swap_contact_monitoring.py` 统一使用：
- env 侧 import：`from .utils.logging import swap_contact_monitoring as swapContact`
  - 位置：
    - `src/robomme/robomme_env/ButtonUnmaskSwap.py:33`
    - `src/robomme/robomme_env/VideoUnmaskSwap.py:32`
    - `src/robomme/robomme_env/VideoRepick.py:33`
- wrapper 侧 import：`from ..robomme_env.utils.logging import swap_contact_monitoring as swapContact`
  - 位置：`src/robomme/env_record_wrapper/RecordWrapper.py:49`

当前源码里实际用到的对外接口有 4 个：
- `swapContact.new_swap_contact_state()`
  - 调用位置：
    - `src/robomme/robomme_env/ButtonUnmaskSwap.py:163`
    - `src/robomme/robomme_env/VideoUnmaskSwap.py:164`
    - `src/robomme/robomme_env/VideoRepick.py:151`
  - 用途：env 构造时创建空的 `SwapContactState`
- `swapContact.reset_swap_contact_state()`
  - 调用位置：
    - `src/robomme/robomme_env/ButtonUnmaskSwap.py:451`
    - `src/robomme/robomme_env/VideoUnmaskSwap.py:440`
    - `src/robomme/robomme_env/VideoRepick.py:316`
  - 用途：env reset 时清空本轮 episode 开始前的接触累计状态
- `swapContact.detect_swap_contacts()`
  - 调用位置：
    - `src/robomme/robomme_env/ButtonUnmaskSwap.py:715`
    - `src/robomme/robomme_env/VideoUnmaskSwap.py:618`
    - `src/robomme/robomme_env/VideoRepick.py:587`
  - 用途：step 结束后，在 swap 时间窗内持续扫描 actor pair，并把接触信息累计到 `SwapContactState`
- `swapContact.get_swap_contact_summary()`
  - 调用位置：`src/robomme/env_record_wrapper/RecordWrapper.py:1586`
  - 用途：close/flush 时把累计 state 压成结构化 summary，供 object log 写入 `collision_events`

这说明 `swap_contact_monitoring.py` 本身不负责落盘：
- env 侧负责创建、清空、实时累计 `SwapContactState`
- wrapper 侧只在 episode 结束时读取 summary，并决定是否写入 object log

## 0.2 `RecordWrapper.py` 用到了哪些接口

`RecordWrapper.py` 统一使用：
- import：`from ..robomme_env.utils.logging import object_log as objectlog`
  - 位置：`src/robomme/env_record_wrapper/RecordWrapper.py:50`

实际调用的对外接口集中在 `_episode_object_log_flush()`：
- `objectlog.flush_episode_log()`
  - 调用位置：`src/robomme/env_record_wrapper/RecordWrapper.py`
  - 用途：把 env 上的 `_episode_object_log_state` 和 wrapper 自己的 `env_id/episode/seed` 组装成最终 record；若 close 阶段拿到有效 contact summary，则直接补进最终 record 的 `collision_events`，然后追加写入 `episode_object_logs.jsonl`

另外两个相关但不是 helper 调用的位置：
- reset 时把 `_episode_object_log_flushed` 置回 `False`：`src/robomme/env_record_wrapper/RecordWrapper.py:721`
- close 时触发 `_episode_object_log_flush()`：`src/robomme/env_record_wrapper/RecordWrapper.py:1639`

## 0.3 各 env 用到了哪些接口，在哪里调用

目前源码里明确接入 `object_log.py` 的 env 只有两个：
- `ButtonUnmaskSwap`
- `VideoUnmaskSwap`

### `ButtonUnmaskSwap`

统一 import：
- `from .utils.logging import object_log as objectlog`
  - 位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:38`

用到的接口：
- `objectlog.init_episode_log()`
  - 调用位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:168`
  - 用途：env 构造时初始化空日志 state
- `objectlog.record_object()`
  - 调用位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:475`
  - 用途：在 `_initialize_episode()` 中通过边界 event 写入 `object_events[event]`，并记录 bin / cube / target cube 的初始快照
- `objectlog.record_swap()`
  - 间接调用封装：`src/robomme/robomme_env/ButtonUnmaskSwap.py:189`
  - 真正 helper 调用：`src/robomme/robomme_env/ButtonUnmaskSwap.py:196`
  - 实际触发位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:658`
  - 用途：某个 swap slot 真正 resolve 出对象对时，记录 `swap_index/object_a/object_b`

### `VideoUnmaskSwap`

统一 import：
- `from .utils.logging import object_log as objectlog`
  - 位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:38`

用到的接口：
- `objectlog.init_episode_log()`
  - 调用位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:169`
  - 用途：env 构造时初始化空日志 state
- `objectlog.record_object()`
  - 调用位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:464`
  - 用途：在 `_initialize_episode()` 中通过边界 event 写入 `object_events[event]`，并记录 bin / cube / target cube 的初始快照

## 0.4 `object_log.py` 当前 schema

当前 episode state 和最终 JSONL record 使用如下结构：
- `object_events: dict[str, dict[str, Any]]`
  - 用于保存对象相关事件，key 是 event 名
  - 同名 event 再次写入时会覆盖旧 payload
  - `reset` / `init` 属于边界事件，写入前会先清空整份 state
- `swap_events: list[dict]`
- `collision_events: list[dict]`

当前两个 swap env 在边界 event 的 payload 中都会记录：
- `bin_list`
- `cube_list`
- `target_cube_list`

这些列表都由 env 在外部先整理成干净的纯 Python 结构，例如：
`{"name": ..., "position": [x, y, z] | None, "color": ...}`
- `objectlog.record_swap()`
  - 间接调用封装：`src/robomme/robomme_env/VideoUnmaskSwap.py:196`
  - 真正 helper 调用：`src/robomme/robomme_env/VideoUnmaskSwap.py:203`
  - 实际触发位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:558`
  - 用途：某个 swap slot 真正 resolve 出对象对时，记录 `swap_index/object_a/object_b`
