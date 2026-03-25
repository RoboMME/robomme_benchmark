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

