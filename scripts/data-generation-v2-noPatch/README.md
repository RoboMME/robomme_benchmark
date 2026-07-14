# RoboMME No-Patch 数据生成

本 README 只说明完整 16 × 9 如何生成、生成器调用的三个独立模块、主要参数，以及最终产物位置。

## 完整 16 × 9 生成

在仓库根目录运行。输出目录必须位于仓库内，并且在运行前不存在或为空。

~~~bash
cd /data/hongzefu/robomme_benchmark-restore-DataGen

uv run --locked scripts/data-generation-v2-noPatch/generate_dataset.py \
  --output-dir artifacts/generated/no-patch-full-16x9-<run-id> \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1
~~~

这会对固定 16 个环境分别生成 episode_0 到 episode_8，共 144 条轨迹。<run-id> 请替换为新的名称，避免复用已有输出目录。

## 生成后的三个调用逻辑

generate_dataset.py 负责生成和合并。每个 task 的 HDF5 与 metadata 写完后，它在同一个 Python 进程中直接调用下面三个独立模块：

~~~text
generate_dataset.py
    ├── validate_generated_dataset_contract.py
    │   └── validate_generated_dataset_contract(...)
    ├── compare_joint_actions.py
    │   └── compare_joint_actions(...)
    └── write_generation_report.py
        └── write_generation_report(...)
~~~

1. validator 复核生成数据、当前 train metadata 和官方参考数据的固定合约。
2. comparator 将生成数据的 action/joint_action 与 data/robomme_data_h5 逐元素比较，默认最大绝对差阈值为 1e-8。
3. writer 汇总前两步的结果，写入 JSON 和 Markdown 报告。

三者都是同进程函数调用，不会调用旧 scripts/data-generation/。如果生成或生成后审计失败，只要输出目录已创建，writer 仍会写出失败报告。

## 参数

完整 16 × 9 命令使用的参数如下：

| 参数 | 本次完整运行的值 | 含义 |
| --- | --- | --- |
| --output-dir | artifacts/generated/no-patch-full-16x9-<run-id> | 生成数据目录；必须在仓库内，且不存在或为空。 |
| --env | all | 固定全部 16 个环境。 |
| --episodes | 9 | 每个环境生成 episode_0 到 episode_8。 |
| --workers | 9 | 最多同时运行 9 个生成 worker。 |
| --gpus | 0,1 | worker 在 GPU 0 和 GPU 1 间轮转分配。 |

生成器固定使用当前 train metadata 的原始 seed 和 difficulty；每个 episode 只尝试一次。

## 对已有完整输出重新比较

已有数据不需要重新生成。下面命令只读复核已有 16 × 9 输出，重新执行 validator 和 comparator，然后刷新中央报告：

~~~bash
uv run --locked scripts/data-generation-v2-noPatch/write_generation_report.py \
  --output-dir artifacts/generated/no-patch-full-16x9 \
  --env all \
  --episodes 9 \
  --workers 9 \
  --gpus 0,1 \
  --max-abs-diff 1e-8
~~~

--workers 和 --gpus 在这个复核命令中只记录到报告参数，不会启动 worker。

## 最终产物

完整运行的生成输出位于 --output-dir：

~~~text
artifacts/generated/no-patch-full-16x9-<run-id>/
├── record_dataset_<Task>.h5
└── record_dataset_<Task>_metadata.json
~~~

完整 16 × 9 会产生 16 个 HDF5 和 16 个 metadata JSON。临时 worker 文件会在结束时清理。

权威比较报告固定写入本目录，而不是写入生成输出目录：

~~~text
scripts/data-generation-v2-noPatch/reports/
├── no_patch_generation_report.json
└── no_patch_generation_report.md
~~~

JSON 保留完整结果；Markdown 给出摘要。该目录只保存最新一次生成或复核的报告，新的运行会覆盖同名 JSON 和 Markdown。

完整 16 × 9 的当前验收结果为：官方和生成最终完成均为 144/144，比较覆盖 73,907 个 joint vectors 与 591,256 个元素，最大绝对差为 5.661269342205344e-09，小于 1e-8。
