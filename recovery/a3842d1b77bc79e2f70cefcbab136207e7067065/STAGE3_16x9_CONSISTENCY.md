# 第三阶段报告：16 个环境 × 9 个 episode 重新生成与一致性审查

## 验收结论

本轮目标范围为 16 个官方 train 环境的 `episode_0`–`episode_8`，即每个环境
9 条、共 144 条 demonstration。恢复链固定使用候选 commit
`a3842d1b77bc79e2f70cefcbab136207e7067065`、该提交的 `uv.lock`、同一 tree 的
`src/robomme/env_metadata/train/`、metadata 原始 seed、单次 seed 尝试、
`pd_joint_pos`、9 workers 和 GPU 0/1。

按用户最后确认的口径——允许 joint action 的小数量级差异，核心要求是同一原 seed
能够再次生成成功——本轮 16×9 范围验收通过，第三阶段可标记为完成。这个结论只表示
“生成契约、结构/元数据、已复测 seed 的当前环境可重复性以及行为回放成功口径”，
不表示字节级或严格内容一致。官方参考数据在整个过程中只读，未被覆盖或混入生成目录。

| 层级 | 结果 | 证据与限定 |
| --- | --- | --- |
| 文件集合 | 通过 | 16 个预期 HDF5 全部存在，无缺失、额外任务文件、生成侧额外 episode、符号链接或临时文件；另有 16 个 metadata JSON |
| 字节级一致 | 不通过 / 不主张 | 官方文件含 100 episode，而本轮文件含 9 episode；且被选 episode 的 canonical 内容摘要也不全相同 |
| HDF5 结构 | 通过 | 144 个 episode 的对象路径、group/dataset/attribute、层级、dtype、shape、存储属性和轨迹长度无差异 |
| metadata/setup | 通过 | task、episode、seed、difficulty、task goal、action space、camera/environment setup 的值、shape、dtype 均精确一致 |
| 严格内容 | 不通过 | 1,996,551 个叶记录中有 5,963 个 `dataset_content` 差异；比较器如实返回退出码 1、`strict_equal=false`、`accepted=false` |
| joint-action 容差 | 大部分通过，PatternLock episode 1 超出原窄策略 | 5,355 个 joint-action 叶存在差异；5,271 条通过 `rtol=1e-7, atol=0, max_abs<=1e-12`，84 条未通过；全局最大绝对差 `5.661269342205344e-9` |
| 当前恢复链同 seed 可重复性 | 通过 | BinFill seed 4000 的两次有效 smoke 严格相同；PatternLock seed 15001/15100 的正式产物与独立重跑严格相同，均为 attempt 1/1 成功 |
| 行为回放 | 通过 | 16 worker 全部退出 0；144/144 `success`、0 step error、0 failure；144 个非空 MP4 |
| 测试 | 部分通过，不掩盖失败 | `tests/dataset/` 31/31 通过；`tests/lightweight/` 136/139 通过，3 项是早于本次恢复修改的历史测试/实现漂移 |

## 正式生成

执行命令：

```bash
scripts/run_recovered_dataset_generator.sh \
  --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8 \
  --episodes 9 \
  --max-workers 9 \
  --gpus 0,1 \
  --save-video
```

退出码为 0。运行器在仓库内隔离 worktree 中应用并逐字校验固化补丁，以候选自己的
锁文件和 Python 3.11.14 启动；输出目录在运行前必须不存在或为空。日志包含 144 次
`attempt 1/1` 和 144 次 `[SUCCESS]`，没有 seed 递增重试。生成契约报告
`passed=true`：16 env、144 episode、73,907 个连续 timestep、0 error，144 条最终
`is_completed=True` 且 `simple_subgoal=All tasks completed`。

正式输出路径：

`artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`

生成 HDF5 共 49,271,290,156 字节。每个 HDF5 和 metadata JSON 的独立 SHA-256
分别保存在：

- `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/generated_h5_sha256.txt`
- `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/generated_metadata_sha256.txt`

两份清单各 16 行。完整生成日志和契约报告为同一报告根目录下的
`generation_driver.log` 与 `generation_contract.json`。

## 官方参考逐叶比较

可复现命令：

```bash
uv run scripts/compare_generated_dataset.py \
  --reference-dir data/robomme_data_h5 \
  --generated-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8 \
  --episodes 0 1 2 3 4 5 6 7 8 \
  --rtol 1e-7 \
  --atol 0 \
  --allow-joint-action-allclose \
  --joint-action-max-abs-diff 1e-12 \
  --report-dir artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/comparison
```

退出码为 1，这是预期的真实审计结果，不应改写成通过。比较范围共 16×9=144 条、
1,996,551 个 dataset 叶；文件集合、对象结构、metadata 和 setup 差异均为 0，内容差异
总数为 5,963：

- 5,355 条位于 `action/joint_action`。RouteStick 的 2,124 条全部是 joint action，
  最大绝对差 `1.1102230246251565e-15`；除 PatternLock 外其他任务的 joint 差异也均
  处于约 `1e-18`–`1e-17` 数量级。
- PatternLock 共 2,083 条差异，其中 1,475 条为 joint action，最大绝对差
  `5.661269342205344e-9`。
- 仅 PatternLock `episode_1` 有非 joint 差异，共 608 个叶：EEF action/state、joint
  state、wrist camera extrinsic，以及稀疏 RGB/depth 像素。浮点状态最大绝对差
  `4.76837158203125e-7`；RGB 涉及 38 个叶、27,342 个通道元素，最大通道差 103；
  depth 涉及 21 个叶、125 个元素，最大差 15。对象集合、126-step 轨迹长度、终止
  信息和完成标志仍精确一致。
- PatternLock episode 1 的数值差异从 timestep 0 已出现，渲染差异从后续 timestep
  才出现。证据支持“数值微差在仿真中向状态和渲染传播”的解释，但不能证明唯一根因
  就是 joint action，因此只作为可能原因记录。

逐条差异、逐叶结果和聚合摘要分别保存在：

- `comparison/differences.jsonl`：全部 5,963 条差异的参考值/生成值、路径、误差和策略结论；
- `comparison/leaf_comparisons.jsonl`：全部 1,996,551 个叶记录；
- `comparison/difference_summary.json`：按 task、episode、字段族汇总；
- `comparison/comparison.json`：完整文件/schema/metadata/content 结论。

`comparison.json` 的 SHA-256 为
`3ce5a5f6f7e7c4a4586f7f9283496b765e7bd9bbffbf091ac0acdddba037b863`，
`differences.jsonl` 的 SHA-256 为
`40994321748cf3cf6e5214fff789e89bf4ff025b12dc96eb5b96a603b915de8c`。

## 同 seed 重复生成

恢复链的重复性不是从 help 或日志推断，而是做了两组独立重跑：

1. BinFill `episode_0`、seed 4000：两次有效的单任务、单 episode、单 worker 运行均
   attempt 1/1 成功；14,858 个 dataset 叶严格相等、差异 0，canonical SHA-256
   都是 `13b2dbdf23a2cfae3b515d6d6be7d34704bf99fb88a4c4e2905f52c33ffa1504`。
2. PatternLock `episode_0/1`、seed 15001/15100：在全新输出目录独立重跑，两条均
   attempt 1/1 成功并完成 96/126 timestep；与本次正式生成相应 episode 的 6,006
   个叶记录在 `rtol=0, atol=0` 下严格相等、差异 0。

PatternLock 重跑日志和严格比较报告位于：

- `artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/runner_patternlock_repeat_episodes_0_1.log`
- `artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/patternlock-formal-vs-repeat-strict/comparison.json`

因此可以确认当前恢复环境对已复测 seed 是确定且可重复的。官方发布数据与当前环境的
差异来自两次不同历史运行之间，这是由“当前正式产物与当前重跑严格相同、但与官方有
差异”推得的范围性结论；无法从现有数据唯一还原官方生成机的 CPU/数值库细节。

## 行为回放

在独立工作目录中执行等价命令（这样生成的回放视频不会混入仓库根目录已有视频）：

```bash
ROOT="$PWD"
REPLAY_RUN="$ROOT/artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/behavior_replay"
cd "$REPLAY_RUN"
UV_CACHE_DIR="$ROOT/.cache/uv" uv run --project "$ROOT" \
  "$ROOT/scripts/dataset_replay.py" \
  --h5-data-dir "$ROOT/artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8" \
  --action-space-type joint_angle \
  --replay-number 9 \
  --replay-log-dir "$REPLAY_RUN/replay_logs"
```

退出码为 0。16 个 task worker 全部 ready 且进程退出码为 0；请求 144 条、实际回放
144 条，全部 outcome 为 `success`，`step_error=0`、`failures=[]`。隔离视频目录包含
144 个非空 MP4，总计 125,986,178 字节。PatternLock episode 1 也成功回放，因此其
稀疏内容差异不改变本轮用户指定的行为验收结果。

汇总和驱动日志：

- `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/behavior_replay/replay_logs/replay_summary.json`
- `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/behavior_replay/dataset_replay_driver.log`

## 测试

所有 Python 命令前均检查 `command -v uv`、根目录 `pyproject.toml` 和 `uv.lock`，
并只使用 uv：

```bash
uv run --locked --extra dev python -m pytest tests/lightweight/
uv run --locked --extra dev python -m pytest tests/dataset/
```

- `tests/lightweight/`：退出码 1，136 passed、3 failed、802 warnings，454.57 秒。
  三项失败均不是本次恢复改动引入，相关源码与测试在本次工作树 diff 中均未修改：
  - 未知环境测试仍期望 `[""]`，而 2026-03-07 的当前实现已改为 `[]`；未知环境不在
    本轮 16 env 范围。
  - SwingXtimes 测试期望 `back and forth`，当前文案是 `back-and-forth`；本轮 HDF5
    `task_goal` 差异为 0。
  - step-error AST 测试仍要求 `DemonstrationWrapper.step()` 内层捕获，但
    2026-02-27 已把统一捕获移到 builder 注入的 `FailAwareWrapper`；正式回放 0 step
    error，dataset 的 16 环境 ee-pose error handling 测试均通过。
- `tests/dataset/`：退出码 0，31 passed、369 warnings，344.45 秒。

日志位于报告根目录的 `tests/lightweight_pytest.log` 和 `tests/dataset_pytest.log`。
轻量测试不能写成“全通过”；三项历史失败是已记录、非本轮阻塞的技术债。

## 最终定性

本轮满足用户修订后的完成条件：16 个环境、每个 9 条、全部使用原 metadata seed
一次生成成功；输出契约完整；结构和元数据精确一致；已复测的 BinFill/PatternLock
seed 独立重跑成功且与当前正式产物严格相同；144/144 行为回放成功。所有已发现差异
都有逐条报告、聚合范围、可能原因、阻塞性与后续动作。没有把这三条 seed 的第二次
重跑结果泛化为“全部 144 条都已生成两次”。

仍必须保留的限定是：官方参考与重建数据不是字节级一致，也不是严格内容一致；原先仅
允许极小 joint-action 差异的窄比较策略返回 `accepted=false`。本阶段之所以完成，依据
是用户随后明确采用“joint action 小数量级差异可接受、同 seed 再次生成成功即可”的
行为/复现性验收口径，而不是把失败的严格比较重新解释成通过。
