# RoboMME No-Patch 数据生成、验证与对比

本目录提供一套不依赖旧 data-generation 目录的独立数据生成链。它只使用当前分支的 src/robomme，并将生成、合约验证、joint_action 数值比较和完整报告串联起来。

生成器不会修改或 monkeypatch src/robomme；planner 的失败回退仅在本目录生成脚本内部实现。所有 Python 命令都必须通过锁定的 uv 环境运行。

## 文件与职责

| 文件/目录 | 职责 |
| --- | --- |
| generate_dataset.py | 生成 HDF5、合并 worker 临时文件、写 metadata，并在同一 Python 进程执行合约验证、joint_action 比较和报告写入。 |
| validate_generated_dataset_contract.py | 只读验证生成数据、官方 HDF5 和当前 train metadata 的固定合约；结果以 JSON 输出到 stdout。 |
| compare_joint_actions.py | 只读逐 timestep、逐元素比较生成数据和官方数据的 action/joint_action；结果以 JSON 输出到 stdout。 |
| write_generation_report.py | 只读复核已有输出，组合验证与比较结果，并写入完整 JSON/Markdown 报告。 |
| reports/ | 固定中央报告目录，保存最新一次生成或复核的权威报告。 |

## 前置条件

请从仓库根目录运行命令：

~~~bash
cd /data/hongzefu/robomme_benchmark-restore-DataGen
command -v uv
test -f pyproject.toml
test -f uv.lock
~~~

运行前必须具备以下仓库内输入：

- 当前分支的 src/robomme。
- 严格 train metadata：src/robomme/env_metadata/train。
- 只读官方参考 HDF5：data/robomme_data_h5。
- 可用于生成的 GPU；生成器通过 --gpus 指定 GPU 编号。
- uv 与根目录的 pyproject.toml、uv.lock。

所有示例都使用 uv run --locked。不要使用裸 python、python3 或 pip。

## 快速开始：生成数据

### 1. 单环境、单 episode smoke

输出目录必须位于仓库内，且在运行前不存在或为空。下面的 run-id 请替换为新的名称。

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/generate_dataset.py \
  --output-dir artifacts/generated/no-patch-smoke-<run-id> \
  --env BinFill \
  --episodes 1 \
  --workers 1 \
  --gpus 0
~~~

成功时 stdout 会输出 status、中央报告路径和 max_abs_diff。该命令会生成 BinFill 的 episode_0，并自动完成验证和官方 joint_action 比较。

### 2. 完整 16 × 9 生成

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/generate_dataset.py \
  --output-dir artifacts/generated/no-patch-full-16x9-<run-id> \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1
~~~

all 表示固定的 16 个环境；episodes 9 表示每个环境生成 episode_0 到 episode_8。不要复用已经包含文件的输出目录。

### 3. 参数语义

| 参数 | 含义 |
| --- | --- |
| --output-dir | 必填。仓库内、不在 data/robomme_data_h5 内、不是符号链接，且必须不存在或为空。 |
| --env / --environment | all 或下列固定任务的逗号分隔无重复子集。 |
| --episodes | 每环境生成/审计的数量，只能为 1 到 9，且范围始终从 episode_0 开始。 |
| --workers / --max-workers | spawn worker 数；实际并发数不会超过任务数。 |
| --gpus / --gpu | 逗号分隔 GPU 编号，例如 0 或 0,1；任务按提交顺序轮转分配。 |

固定任务顺序如下：

~~~text
PickXtimes
StopCube
SwingXtimes
BinFill
VideoUnmaskSwap
VideoUnmask
ButtonUnmaskSwap
ButtonUnmask
VideoRepick
VideoPlaceButton
VideoPlaceOrder
PickHighlight
InsertPeg
MoveCube
PatternLock
RouteStick
~~~

## 生成过程与固定策略

每个请求范围内的 task/episode 都执行以下过程：

1. 严格读取当前 src/robomme/env_metadata/train 中对应记录的 task、episode、seed 和 difficulty；metadata 文件集合、记录数和 episode 0 到 99 均会被检查。
2. 用 metadata 的原始 seed 和 difficulty 创建 gym 环境，并以 RobommeRecordWrapper 录制。
3. episode_0 到 episode_2 启用 z failure recovery；episode_3 到 episode_5 启用 xy recovery；episode_6 及之后不启用 recovery。
4. 选择当前 Panda Arm 或 Panda Stick planner，执行环境 task_list。screw 规划最多尝试 3 次，之后 RRTStar 最多尝试 3 次；这仅是脚本内的 planner 子类逻辑。
5. 每个 episode 只尝试一次原始 seed；不会使用 seed+1 重试。
6. worker 将临时 HDF5 写入 <output-dir>/.workers/，成功后按 task 合并为最终 HDF5；完成或失败时临时 worker 目录会清理。
7. 生成完成后在同一进程调用独立 validator、comparator 和 writer；任何合约或阈值失败都会令生成命令以退出码 1 结束。

生成器直接导入当前 src/robomme 所需环境、wrapper、planner 与异常类型；不会读取、导入、调用或复制旧 scripts/data-generation/。

## 新生成数据的产物

对每个请求 task，最终 <output-dir> 包含：

~~~text
<output-dir>/
├── record_dataset_<Task>.h5
└── record_dataset_<Task>_metadata.json
~~~

完整 16 × 9 运行会有 16 个 HDF5 和 16 个 metadata JSON。metadata JSON 的结构为：

~~~json
{
  "env_id": "BinFill",
  "record_count": 9,
  "records": [
    {
      "task": "BinFill",
      "episode": 0,
      "seed": 4000,
      "difficulty": "easy"
    }
  ]
}
~~~

HDF5 中每个 episode 使用如下关键层级：

~~~text
episode_<i>/
├── setup/seed
├── setup/difficulty
└── timestep_<j>/
    ├── action/joint_action
    └── info/is_completed
~~~

新的输出目录不会写 reports/ 子目录。过去 artifact 目录中如果仍有报告文件，它们是历史副本，不是当前 writer 的写入目标。

## 自动与独立验证

### 生成后自动执行

generate_dataset.py 会在 HDF5 合并后自动执行以下两项审计，并且只有两项都通过才返回成功：

- 数据合约验证。
- 与 data/robomme_data_h5 的 joint_action 逐元素比较。

### 单独执行数据合约验证

以下命令只读 HDF5/metadata，将结构化 JSON 输出到 stdout；成功退出码为 0，失败为 1。

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/validate_generated_dataset_contract.py \
  --output-dir artifacts/generated/no-patch-full-16x9 \
  --env all \
  --episodes 9 \
  --metadata-root src/robomme/env_metadata/train \
  --reference-root data/robomme_data_h5
~~~

它严格检查：

- 请求 episode 的数值 timestep 必须是连续的 timestep_0 到 timestep_N。
- setup/seed 与 setup/difficulty 必须和当前 train metadata 对齐。
- 每个 action/joint_action 必须是 shape (8,) 的 float64，且所有数值有限。
- 最后一个数值 timestep 必须存在严格 bool 标量 info/is_completed。
- 生成 HDF5 的 episode 集合必须精确等于请求范围；官方 HDF5 必须包含请求范围，但允许额外的 episode_9 到 episode_99。
- 生成 metadata JSON 必须与请求的 train metadata 记录精确一致。
- 官方和生成轨迹的最终完成数都必须等于请求轨迹数。

### 单独比较 joint_action

以下命令只读两侧 HDF5，将结构化 JSON 输出到 stdout；成功退出码为 0，失败为 1。

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/compare_joint_actions.py \
  --output-dir artifacts/generated/no-patch-full-16x9 \
  --env all \
  --episodes 9 \
  --reference-root data/robomme_data_h5 \
  --max-abs-diff 1e-8
~~~

比较器对每个选定 task、episode、timestep 和 joint_action 元素逐一比较，并额外检查 HDF5 路径、连续 timestep、shape、dtype 和有限数值。结果包含 joint vector/element 覆盖数、不同元素数、最大绝对差、最大差位置和阈值结论。

不同元素可以存在；通过条件不是字节级或逐元素严格相等，而是无结构/数值错误且最大绝对差不超过 --max-abs-diff。

## 只读复核并写完整报告

对已有输出重新执行完整验证和比较时，使用：

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/write_generation_report.py \
  --output-dir artifacts/generated/no-patch-full-16x9 \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1 \
  --metadata-root src/robomme/env_metadata/train \
  --reference-root data/robomme_data_h5 \
  --max-abs-diff 1e-8
~~~

该命令不会创建 gym 环境、不会启动生成 worker，也不会修改 HDF5 或 metadata；它只读复核已有输出，然后刷新中央报告。--workers 和 --gpus 在此 CLI 中只写入报告参数，不控制并发。

可选的 --prior-report 只能传入属于同一个 --output-dir 的、此前保留的生成报告 JSON。它用于保留原始 worker/provenance 信息；没有匹配的 JSON 时应省略该参数。

## 中央报告与字段说明

无论报告来自生成成功、生成失败（writer 可写时）还是已有输出复核，权威报告固定写入：

~~~text
scripts/data-generation-v2-noPatch/reports/
├── no_patch_generation_report.json
└── no_patch_generation_report.md
~~~

这对固定文件只保留最新一次运行的报告。不要并发运行多个生成或复核命令；新的运行会覆盖旧 JSON 和 Markdown。若需要区分历史运行，请在下一次运行前自行保存中央报告副本。

JSON 使用 schema_version 2，并保留完整逐轨迹审计。主要字段为：

| 字段 | 内容 |
| --- | --- |
| current_head、uv_lock_sha256、generated_at_utc | 当前代码来源、锁文件哈希与时间。 |
| parameters | 输出目录、任务、episode、GPU、worker、metadata/reference 路径与阈值。 |
| generation | 请求/成功/失败计数及逐 worker 结果；已有输出复核可保留先前 provenance。 |
| validation.scope | 任务、episode 范围、期望轨迹数和是否为完整 16 × 9。 |
| validation.metadata | 每个生成 metadata JSON 的审计与错误。 |
| validation.generated、validation.official | 文件数、episode 数、最终完成数、joint 覆盖量和逐轨迹审计。 |
| validation.joint_action_comparison | 比较错误、不同元素数、最大差和位置、阈值及通过结论。 |
| validation.acceptance | 官方/生成完成数与最大差的验收摘要。 |
| report_paths | JSON 与 Markdown 的绝对路径。 |

Markdown 是人类可读摘要；完整错误、每条轨迹的 timestep/终态和比较详情保留在 JSON。

## 当前完整 16 × 9 结果

当前中央报告可直接查看：

- reports/no_patch_generation_report.json
- reports/no_patch_generation_report.md

截至 2026-07-14，中央报告的完整 16 × 9 只读复核结果为：

| 项目 | 结果 |
| --- | --- |
| 范围 | 16 个任务 × episode_0 到 episode_8 = 144 条轨迹。 |
| 报告状态 | passed。 |
| 官方最终完成 | 144/144。 |
| 生成最终完成 | 144/144。 |
| metadata、生成 HDF5、官方 HDF5 错误 | 均为 0。 |
| joint vectors | 73,907。 |
| joint elements | 591,256。 |
| 不同元素数 | 16,380；仅作统计，不是失败条件。 |
| 最大绝对差 | 5.661269342205344e-09。 |
| 阈值 | 1e-8。 |
| 最大差位置 | PatternLock / episode_1 / timestep_121 / element_6。 |
| 结论 | 通过；最大差小于阈值。 |

该结果不声称字节级一致、所有 action 字段一致或行为回放一致；它只说明本目录定义的 HDF5/metadata 合约和选定范围内的 action/joint_action 数值比较均已通过。

## 常见限制与排错

- 输出目录不在仓库内、位于参考数据内、是符号链接或非空时，生成器会拒绝运行。
- --episodes 不能大于 9；本脚本的验收范围固定为每环境从 episode_0 开始的前 1 到 9 条。
- --env 只能使用固定 16 任务的 all 或无重复子集；未知任务会直接失败。
- validator 和 comparator 只向 stdout 写 JSON，不产生独立报告文件；完整 JSON/Markdown 由 writer 统一写入中央 reports/。
- 参考 data/robomme_data_h5 始终只读，生成输出必须使用独立的 artifacts/generated/<run-id> 目录。
- 报告中的 current_head 和 uv.lock SHA-256 是运行时记录的 provenance；重跑后数值与时间可能变化。
- 如果要仅检查已有输出，请使用 write_generation_report.py，而不是再次调用生成器。
