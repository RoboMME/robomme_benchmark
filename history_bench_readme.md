# HistoryBench

面向 ManiSkill 桌面任务的数据集生成与评估工具。

---

## ✅ 目前支持的任务（Solve_3 parallel.py）
```
HighlightPickube
PickXtimes
PickXobjBin
PutOntoBefore
PutOntoAfter
ButtonChooseBin
VideoChooseBin
```
> 任务名以脚本内的注册名为准。

---

## 📀 数据集生成（Solve_3 parallel.py）
`Solve_3 parallel.py` 会生成**完整数据集**，包含：
- **setup**：
  - `language_goal`（完整任务的自然语言指令）
  - `seed`
  - `task_list`（该 seed 下的子任务序列）
- **timestep**（逐步记录）：
  - `action`
  - `image`
  - `wrist_image`
  - `state`
  - `current_task_demonstration`（bool）
  - `current_task_index`
  - `current_task_name`（该步对应子任务的自然语言指令）

语义约定：
- 当 `current_task_demonstration = True` 时，说明该 timestep 属于**视频演示**的一部分。
- `current_task_name` 即该子任务的自然语言描述。
- **随机性**（物体朝向、子任务采样等）**仅由 `seed` 唯一决定**，相同 seed 可复现实验。

---

## 🧪 评估（Evaluate_policy）
评估流程：
1. 先调用 `get_demonstration_trajectory()`，在给定 `seed` 下执行并**完成视频演示**；
2. 演示完成后，开始使用 `step()` 对策略进行 evaluate。

伪代码示例：
```python
# 1) 获得并执行演示
traj = get_demonstration_trajectory(seed=SEED)
# （内部会根据 seed 复现环境并完成演示）

# 2) 评估策略
obs = env.reset(seed=SEED)
while not done:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    # 记录评估指标
```

> 以上仅为流程说明；请以你实现的函数签名与返回值为准。

---

## 🗂️ 数据字段速览
- **setup**：`language_goal`, `seed`, `task_list`
- **timestep**：`action`, `image`, `wrist_image`, `state`, `current_task_demonstration`, `current_task_index`, `current_task_name`

---

如需增删字段或扩展任务，请在脚本中按上述约定更新相应记录逻辑。

