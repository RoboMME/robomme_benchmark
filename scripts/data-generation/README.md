# RoboMME dataset generation

本目录集中保存 `dataset-gen` 分支的 dataset 生成、完整性验证和 16×9 对比工具。
根目录的 `AGENTS.md` 是仓库级执行账本，不属于本目录。

## 工具和固定版本

- `generate_dataset.sh`：恢复固定历史生成器并生成 HDF5 dataset。
- `generate_dataset_a3842d1.patch`：将历史生成器改为仓库内路径、固定 train metadata 和原始 seed。
- `validate_generated_dataset_contract.py`：验证文件集合、metadata、HDF5 结构、连续 timestep 和完成标志。
- `compare_joint_actions.py`：只读比较官方 dataset 与重新生成 dataset 的 joint action 和最终完成状态。
- `reports/DATASET_COMPARISON_16x9.md`：最后一次 16×9 对比摘要。

生成入口固定使用候选 commit `a3842d1b77bc79e2f70cefcbab136207e7067065`、候选 `uv.lock` 和 Python 3.11.14。
脚本会在 `artifacts/recovery/` 创建隔离 worktree 和 uv 环境，并在生成结束后自动运行契约验证。

## 前置条件

在仓库根目录执行命令，并确认以下文件存在：

```bash
command -v uv
test -f pyproject.toml
test -f uv.lock
test -d data/robomme_data_h5
```

官方原始 dataset 必须位于 `data/robomme_data_h5/`，不能覆盖该目录。完整生成需要可用 GPU、足够磁盘空间和候选生成器依赖。

## 完整重新生成 dataset

下面的命令生成全部 16 个环境、每个环境 100 个 episode。`--output-dir` 必须是仓库内、候选 commit 专用且不存在或为空的新目录：

```bash
scripts/data-generation/generate_dataset.sh \
  --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/full-train-episodes-0-99 \
  --episodes 100 \
  --max-workers 20 \
  --gpus 1
```

默认环境集合为：

`PickXtimes`、`StopCube`、`SwingXtimes`、`BinFill`、`VideoUnmaskSwap`、`VideoUnmask`、`ButtonUnmaskSwap`、`ButtonUnmask`、`VideoRepick`、`VideoPlaceButton`、`VideoPlaceOrder`、`PickHighlight`、`InsertPeg`、`MoveCube`、`PatternLock`、`RouteStick`。

生成器会固定 train metadata 中的 seed，使用单次 seed attempt，并在生成后检查每个预期环境和 episode 是否完整。生成数据、契约报告、恢复 worktree、uv 环境和缓存都写在仓库内。

## 独立验证完整性

生成入口已经自动调用验证器。需要对已有目录重新验证时，在仓库根目录执行：

```bash
EXPECTED_ENVS=(
  PickXtimes StopCube SwingXtimes BinFill VideoUnmaskSwap VideoUnmask
  ButtonUnmaskSwap ButtonUnmask VideoRepick VideoPlaceButton VideoPlaceOrder
  PickHighlight InsertPeg MoveCube PatternLock RouteStick
)
EXPECTED_ENV_ARGS=()
for env in "${EXPECTED_ENVS[@]}"; do
  EXPECTED_ENV_ARGS+=(--expected-env "$env")
done

uv run --locked scripts/data-generation/validate_generated_dataset_contract.py \
  --generated-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/full-train-episodes-0-99 \
  --metadata-root src/robomme/env_metadata/train \
  --workspace-root . \
  --report artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/full-train-episodes-0-99/generation_contract-rerun.json \
  --expected-episodes 100 \
  "${EXPECTED_ENV_ARGS[@]}"
```

验证通过时会报告 HDF5 文件数、environment 数、episode 数、timestep 数和错误数。报告路径必须位于仓库内，且不能写入 dataset 或 metadata 输入目录。

## 复现最后一次 16×9 对比

当前保留的最终生成 dataset 为：

`artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`

官方参考 dataset 为：

`data/robomme_data_h5/`

使用默认路径重新生成 JSON 和 Markdown 对比结果：

```bash
uv run --locked scripts/data-generation/compare_joint_actions.py
```

默认比较 16 个任务的 episode 0--8。机器报告位于最终报告目录的 `joint_16x9/comparison.json`，Markdown 摘要位于本目录的 `reports/DATASET_COMPARISON_16x9.md`。

当前 16×9 结果：

- 官方和生成数据最终完成状态均为 `144/144`。
- joint action 严格相等为否。
- `5,355` 条 joint 路径、`16,380` 个 joint 元素存在严格数值差异。
- 最大绝对差为 `5.661269342205344e-9`。

因此本结果不能声明 byte-level、全内容严格一致或数值容差通过；joint 差异只做报告，不影响结构和完成状态契约的退出码。

## 当前保留的产物

- 原始 dataset：`data/robomme_data_h5/`
- 原始 dataset 回放：`runs/replay_videos/`
- 最后一次 16×9 生成 dataset：`artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`
- 最后一次 16×9 完整报告：`artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`
- 原始 dataset 审计与回放日志：`artifacts/reports/reference/`

其他 smoke、重复生成、旧候选恢复目录、测试临时目录、uv/cache/venv 和 Python 缓存均属于中间产物，应在本地清理，不作为 dataset 交付物。
