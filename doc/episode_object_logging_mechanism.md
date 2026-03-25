# 当前 object log 机制说明

本文解释 `/runs/replay_videos/episode_object_logs.jsonl` 对应的当前 logging 机制。这里的 logging 不是普通文本 logger，而是一套 episode 级别的结构化 JSONL 输出。当前统一入口模块是 `src/robomme/env_record_wrapper/object_log.py`。

## 0. 最小总结

`object_log.py` 提供的是一组 episode 级结构化日志 helper，统一按模块别名调用：
- env 构造时调用 `objectlog.init_episode_object_log_state()`，准备空 state
- env reset 时调用 `objectlog.record_reset_objects()`，记录初始对象快照
- env step 期间按需调用 `objectlog.append_episode_object_swap_event()`，记录实际发生的 swap 对
- `RecordWrapper.close()` 里调用 `_episode_object_log_flush()`，必要时先补 `objectlog.append_episode_object_collision_event()`，再用 `objectlog.build_episode_object_log_record()` + `objectlog.append_episode_object_log_record()` 写入 `episode_object_logs.jsonl`

`RecordWrapper.py` 在这条链路里的职责很单一：
- 不负责生成对象快照和 swap 事件
- 只负责在 episode 结束时把 env 上已经累积好的 state 收口、补 collision summary、并落盘

## 0.1 `RecordWrapper.py` 用到了哪些接口

`RecordWrapper.py` 统一使用：
- import：`from ..env_record_wrapper import object_log as objectlog`
  - 位置：`src/robomme/env_record_wrapper/RecordWrapper.py:50`

实际调用的对外接口只有 3 个，全部集中在 `_episode_object_log_flush()`：
- `objectlog.append_episode_object_collision_event()`
  - 调用位置：`src/robomme/env_record_wrapper/RecordWrapper.py:1588`
  - 用途：如果 `swap_contact_state` 显示本 episode 确实发生过接触，就把接触摘要补进 `collision_events`
- `objectlog.build_episode_object_log_record()`
  - 调用位置：`src/robomme/env_record_wrapper/RecordWrapper.py:1594`
  - 用途：把 env 上的 `_episode_object_log_state` 和 wrapper 自己的 `env_id/episode/seed` 组装成最终 record
- `objectlog.append_episode_object_log_record()`
  - 调用位置：`src/robomme/env_record_wrapper/RecordWrapper.py:1607`
  - 用途：把最终 record 追加写入 `episode_object_logs.jsonl`

另外两个相关但不是 helper 调用的位置：
- reset 时把 `_episode_object_log_flushed` 置回 `False`：`src/robomme/env_record_wrapper/RecordWrapper.py:721`
- close 时触发 `_episode_object_log_flush()`：`src/robomme/env_record_wrapper/RecordWrapper.py:1639`

## 0.2 各 env 用到了哪些接口，在哪里调用

目前源码里明确接入 `object_log.py` 的 env 只有两个：
- `ButtonUnmaskSwap`
- `VideoUnmaskSwap`

### `ButtonUnmaskSwap`

统一 import：
- `from ..env_record_wrapper import object_log as objectlog`
  - 位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:38`

用到的接口：
- `objectlog.init_episode_object_log_state()`
  - 调用位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:168`
  - 用途：env 构造时初始化空日志 state
- `objectlog.record_reset_objects()`
  - 调用位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:475`
  - 用途：在 `_initialize_episode()` 中记录 bin / cube / target cube 的初始快照
- `objectlog.append_episode_object_swap_event()`
  - 间接调用封装：`src/robomme/robomme_env/ButtonUnmaskSwap.py:189`
  - 真正 helper 调用：`src/robomme/robomme_env/ButtonUnmaskSwap.py:196`
  - 实际触发位置：`src/robomme/robomme_env/ButtonUnmaskSwap.py:658`
  - 用途：某个 swap slot 真正 resolve 出对象对时，记录 `swap_index/object_a/object_b`

### `VideoUnmaskSwap`

统一 import：
- `from ..env_record_wrapper import object_log as objectlog`
  - 位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:38`

用到的接口：
- `objectlog.init_episode_object_log_state()`
  - 调用位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:169`
  - 用途：env 构造时初始化空日志 state
- `objectlog.record_reset_objects()`
  - 调用位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:464`
  - 用途：在 `_initialize_episode()` 中记录 bin / cube / target cube 的初始快照
- `objectlog.append_episode_object_swap_event()`
  - 间接调用封装：`src/robomme/robomme_env/VideoUnmaskSwap.py:196`
  - 真正 helper 调用：`src/robomme/robomme_env/VideoUnmaskSwap.py:203`
  - 实际触发位置：`src/robomme/robomme_env/VideoUnmaskSwap.py:558`
  - 用途：某个 swap slot 真正 resolve 出对象对时，记录 `swap_index/object_a/object_b`

### 没有直接接入的 env

按当前源码搜索结果，其他 env 没有直接 import 或调用 `object_log.py` 里的接口。
这意味着：
- 它们不会主动写 reset object 快照
- 也不会主动写 swap event
- 只有在 wrapper 侧存在可用 state 时，close 阶段才可能统一 flush 出一条 record

## 1. 输出文件是什么

输出文件名固定为 `episode_object_logs.jsonl`，由 `objectlog.append_episode_object_log_record()` 追加写入到 `output_root` 下。

关键实现：
- `EPISODE_OBJECT_LOG_FILENAME = "episode_object_logs.jsonl"`：`src/robomme/env_record_wrapper/object_log.py:19`
- 追加写入 JSONL：`src/robomme/env_record_wrapper/object_log.py:277`

每一行是一个 episode 的完整记录，不是 step 级日志。

当前 `/runs/replay_videos/episode_object_logs.jsonl` 里共有 58 条记录，按 `env` 分布为：
- `ButtonUnmaskSwap`: 20
- `VideoUnmaskSwap`: 20
- `VideoRepick`: 18

当前记录字段是：
- `env`
- `episode`
- `seed`
- `bin_list`
- `cube_list`
- `target_cube_list`
- `swap_events`
- `collision_events`

## 2. 内部 state 长什么样

环境对象上挂了一个 `_episode_object_log_state`，它是一个简单 dict：
- `bin_list`
- `cube_list`
- `target_cube_list`
- `swap_events`
- `collision_events`

初始化逻辑：
- 空 state 定义：`src/robomme/env_record_wrapper/object_log.py:60`
- 挂到 env 上：`src/robomme/env_record_wrapper/object_log.py:92`

这意味着日志状态是“挂在 env 上的 episode 级缓存”，而不是 wrapper 自己维护一份副本。

## 3. 什么时候初始化

### 3.1 环境构造时

在两个 swap 环境的 `__init__` 里，会先创建碰撞监控状态，再初始化 episode object log state：
- `ButtonUnmaskSwap`：`src/robomme/robomme_env/ButtonUnmaskSwap.py:168`
- `VideoUnmaskSwap`：`src/robomme/robomme_env/VideoUnmaskSwap.py:169`

这一步只是准备空容器，不会写文件。

### 3.2 wrapper reset 时

`RecordWrapper.reset()` 会把 `_episode_object_log_flushed` 置回 `False`，确保新 episode 最多 flush 一次：
- `src/robomme/env_record_wrapper/RecordWrapper.py:721`

注意这里并不会重新构造 env 上的 object log state；真正的数据内容刷新发生在环境自己的 `_initialize_episode()`。

## 4. reset 时记录了什么

在两个 swap 环境的 `_initialize_episode()` 中，会做三件事：
1. `reset_swap_contact_state(...)` 清空碰撞统计
2. 根据当前 episode 的场景生成结果，整理 `bin_list` / `cube_list` / `target_cube_list`
3. 调用 `objectlog.record_reset_objects(...)` 写入 env 上的 episode log state

位置：
- `ButtonUnmaskSwap` reset 记录：`src/robomme/robomme_env/ButtonUnmaskSwap.py:475`
- `VideoUnmaskSwap` reset 记录：`src/robomme/robomme_env/VideoUnmaskSwap.py:464`

`objectlog.record_reset_objects()` 做的事情很克制：
- 只保留 actor 的 `name`、世界坐标 `position`、以及外部传入的 `color`
- 同时把 `swap_events` 清空

对应实现：
- actor 序列化：`src/robomme/env_record_wrapper/object_log.py:131`
- reset 对象写入：`src/robomme/env_record_wrapper/object_log.py:163`

所以 `bin_list` / `cube_list` / `target_cube_list` 反映的是“本 episode 初始化后的对象布局快照”。

## 5. swap 事件是怎么记的

swap 不是在 reset 时一次性决定全部细节，而是在 step 里按 slot 动态 resolve。

流程如下：
1. `_refresh_swap_schedule()` 先创建 `(None, None, start_step, end_step)` 形式的时间槽
2. 在 step 中，当 timestep 落进某个 slot 时，调用 `_resolve_swap_schedule_slot(slot_idx)`
3. 这里通过 `select_dynamic_swap_pair(...)` 选出本次交换的两个 bin
4. 一旦 resolve 成功，就调用 `_append_episode_object_swap_event(...)`
5. 最终落到 `objectlog.append_episode_object_swap_event(...)`，只记录：
   - `swap_index`
   - `object_a`
   - `object_b`

位置：
- `ButtonUnmaskSwap` helper 调用：`src/robomme/robomme_env/ButtonUnmaskSwap.py:196`
- `VideoUnmaskSwap` helper 调用：`src/robomme/robomme_env/VideoUnmaskSwap.py:203`
- helper 实现：`src/robomme/env_record_wrapper/object_log.py:187`

因此 `swap_events` 代表的是“实际被调度出来的 swap 对”，而不是原始配置参数。

## 6. collision 是怎么记的

collision 统计分两层：

### 6.1 step 内实时累计

每个 step 结束后，swap 环境会调用 `detect_swap_contacts(...)`：
- `ButtonUnmaskSwap`：`src/robomme/robomme_env/ButtonUnmaskSwap.py:719`
- `VideoUnmaskSwap`：`src/robomme/robomme_env/VideoUnmaskSwap.py:622`

这个函数只在“当前 timestep 落在 swap schedule 时间窗内”时才监控：
- 监控窗口判断：`src/robomme/robomme_env/utils/swap_contact_monitoring.py:59`

它会遍历 `actors` 中的两两组合，调用 `scene.get_pairwise_contact_forces(actor_a, actor_b)`，把结果累计到 `SwapContactState`：
- 首次碰撞 step：`first_contact_step`
- 出现过碰撞的 pair：`contact_pairs`
- 全局最大力：`max_force_norm`
- 最大力对应 pair：`max_force_pair`
- 最大力出现的 step：`max_force_step`
- 每个 pair 的最大力：`pair_max_force`

数据结构定义：`src/robomme/robomme_env/utils/swap_contact_monitoring.py:14`

额外一点：第一次发现某个 pair 碰撞时，还会 `print` 一条终端信息，但这不是 JSONL 的持久化写入：
- `src/robomme/robomme_env/utils/swap_contact_monitoring.py:156`

### 6.2 close 时一次性写入 object log

真正写进 `collision_events` 的时机，不在 step 内，而是在 `RecordWrapper.close()` 触发的 `_episode_object_log_flush()`：
- flush 入口：`src/robomme/env_record_wrapper/RecordWrapper.py:1583`
- `close()` 调用 flush：`src/robomme/env_record_wrapper/RecordWrapper.py:1639`

这里的逻辑是：
1. 从 `env.unwrapped.swap_contact_state` 取累计状态
2. 用 `get_swap_contact_summary(...)` 生成摘要
3. 只有当 `swap_contact_detected == True` 时，才调用 `objectlog.append_episode_object_collision_event(...)`
4. 之后再构造整条 record 并 append 到 JSONL

所以当前机制下：
- 没碰撞时，`collision_events` 是空列表
- 碰撞过时，通常会追加一条 summary event
- 这条 summary 是整个 episode 维度的，不是逐 step 事件流

## 7. 最终 flush 是怎么做的

`_episode_object_log_flush()` 是唯一的结构化落盘入口：
- 构造 record：`src/robomme/env_record_wrapper/object_log.py:252`
- 追加到 JSONL：`src/robomme/env_record_wrapper/object_log.py:277`

record 内容由 env 上缓存的 state + wrapper 上下文组成：
- `env` 来自 `self.env_id`
- `episode` 来自 `self.episode`
- `seed` 来自 `self.seed`
- 其余对象和事件都来自 env 上的 `_episode_object_log_state`

同时有一个防重开关：
- 若 `_episode_object_log_flushed` 已经为 `True`，就直接返回
- 见 `src/robomme/env_record_wrapper/RecordWrapper.py:1584`

这保证了同一个 episode 在 `close()` 流程中只会写一次 JSONL。

## 8. 和普通 logger 的关系

项目里也有普通 `logging`：
- logger 定义：`src/robomme/logging_utils.py:1`

但它和这里的 `episode_object_logs.jsonl` 是两条不同链路：
- `logger.debug(...)` 用于调试信息输出
- `episode_object_logs.jsonl` 是结构化 episode 元数据

换句话说，当前“logging 机制”更准确地说是：
- 一条文本调试日志链路
- 一条 episode 级 JSONL 结构化日志链路

你给的这个文件属于后者。

## 9. 当前机制的特点

优点：
- 结构稳定，适合后处理和统计
- episode 级别写一次，IO 成本低
- 把 reset 布局、实际 swap 对、碰撞摘要放在同一条记录里，便于回放和排查

限制：
- 不是 step 级事件流，无法直接还原每一步发生了什么
- `collision_events` 目前本质上是 episode summary，不是完整时间序列
- 非 swap 环境如果没有显式调用这些 helper，JSONL 里可能只有空字段或根本没有相关信息
- `objectlog.record_reset_objects()` 会清空 `swap_events`，所以它假设每个 episode 的 reset 发生在任何 swap 记录之前

## 10. 结合当前文件怎么理解

对 `/runs/replay_videos/episode_object_logs.jsonl` 来说，每一行可以理解为：
- 这个 episode 一开始有哪些 bin / cube / target cube，以及它们的位置和颜色
- 在运行过程中实际发生了哪些 swap 配对
- 这个 episode 的 swap 期间是否检测到物体接触，以及最大接触力摘要

不是：
- 每一步的完整轨迹
- 每次碰撞发生时的完整时序日志
- logger 控制台输出的落盘结果

## 11. 一句话总结

当前机制是“环境内累积 episode object state，wrapper 在 `close()` 时一次性 flush 为 JSONL 记录”；`episode_object_logs.jsonl` 是 episode 级结构化旁路日志，不是 step 级文本日志。
