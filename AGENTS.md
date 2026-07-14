# RoboMME 数据生成脚本恢复仓库

## 仓库目标

本仓库专门用于寻找、恢复并验证 RoboMME dataset 的生成脚本。最终目标不是只找到一个历史文件，而是完成以下闭环：

1. 下载官方参考 dataset；
2. 从 Git 历史中找到最新且可用的数据生成脚本及其完整依赖；
3. 用恢复后的脚本重新生成数据，并与官方参考 dataset 做一致性审查。

不要把无关的 benchmark 功能开发、模型训练或大规模重构混入本任务。三个阶段必须按顺序执行；上一阶段没有证据证明完成时，不得把下一阶段标记为完成。

## 全局执行规则

- 所有计划、进展、结论和阻塞说明都必须使用中文。
- 每次开始工作前先阅读本文件；每个阶段开始、取得关键进展、遇到阻塞以及完成时，都必须更新本文件中的“当前进度”和“追加式执行日志”。不能只在聊天消息、终端输出或其他报告里记录进展。
- `AGENTS.md` 是本任务的持续状态账本。更新进度表的同时保留已有日志，不得覆盖或删除旧记录。
- 所有下载数据、生成数据、审查产物和日志都必须位于本仓库根目录内。禁止使用仓库外目录作为真实存储位置，也禁止用符号链接、bind mount 或仅存于 `/tmp` 的文件绕过此限制。
- 官方参考数据固定放在 `data/robomme_data_h5/`；重新生成的数据必须放在另一个仓库内目录，例如 `artifacts/generated/<commit>/`，绝不能覆盖或混入参考数据。
- 当前 `.gitignore` 和 `.dockerignore` 没有忽略 `data/`。下载前必须先避免大型数据被 Git 跟踪或被无意加入 Docker build context，并在执行日志中记录具体处理。除非用户明确要求，不得提交下载或生成的 HDF5、图片、视频等大型产物。
- 任何“完成”“一致”或“可用”的判断都必须附带可复现命令、退出状态、输出路径和审查摘要。没有证据时只能写“未验证”或“进行中”。
- 不得用破坏性 Git 操作清理工作区。检查历史优先使用 `git log`、`git show`、`git ls-tree`、`git diff`；需要运行历史版本时使用隔离 worktree 或恢复分支，不能覆盖用户现有修改。

## Python 与 uv 规则

在执行任何 Python 命令前，必须先运行 `command -v uv`，并检查仓库中的 `uv.lock` 或 `pyproject.toml`。

本仓库已经包含 `uv.lock` 和 `pyproject.toml`，因此：

- 运行脚本必须使用 `uv run ...`，不能直接使用 `python` 或 `python3`；
- 安装依赖必须使用 `uv add` 或 `uv pip install`，不能使用 `pip`；
- 创建虚拟环境必须使用 `uv venv`，不能使用 `python -m venv`；
- 测试命令也必须由 `uv run` 启动，例如 `uv run python -m pytest ...`；
- 只要 uv 可用，就绝不能回退到裸 `python`、`python3` 或 `pip`。

## 第一阶段：下载官方参考 dataset

依据根目录 `readme.md`：

- 官方来源：`https://huggingface.co/datasets/Yinpei/robomme_data_h5`
- README 声明的数据规模：16 个任务，共 1,600 条 demonstration，每个任务 100 条；
- 仓库内固定下载目录：`data/robomme_data_h5/`；
- `scripts/dataset_replay.py` 的默认读取目录也是 `data/robomme_data_h5/`。

执行要求：

1. 先记录当前磁盘空间、下载工具、源 URL/版本信息和目标绝对路径。
2. README 只给出了下载链接，没有给出 CLI；实际采用的补充下载命令必须在日志中明确标注为“根据 README 链接补充”，并且仍须由 uv 管理相关 Python 工具。
3. 下载目标必须解析到本仓库下的 `data/robomme_data_h5/`，不能下载到其他位置后再做链接。
4. 下载后至少记录文件清单、文件数量、总大小以及可获得的 revision/checksum 信息，并核对是否符合 README 的 16 个任务、1,600 条 demonstration、每任务 100 条声明。
5. 使用以下回放入口进行初步 sanity check，并记录成功/失败的任务和视频输出位置：

   ```bash
   uv run scripts/dataset_replay.py --h5-data-dir ./data/robomme_data_h5
   ```

只有参考数据已完整落在仓库内、清单核对完成且结果写入本文件后，第一阶段才能标记为“完成”。

## 第二阶段：扫描 Git 并恢复最新可用生成脚本

只能在第一阶段完成后正式开始本阶段。“最新”不等于“可用”，不得只按文件名、提交日期或分支 tip 做结论。

最低扫描范围：

1. 在不破坏工作区的前提下更新并检查所有 branch、remote ref 和 tag；记录扫描时的远端状态和 commit SHA。
2. 搜索包含 `dataset`、`generate`、`record`、`rollout`、`demonstration`、`inspect`、`replay` 等关键词的提交和历史路径，并追踪文件的新增、重命名和删除。
3. 优先核查已知旧入口名：
   - `generate-dataset-control-seed-readJson-advanceV3.py`
   - `generate_dataset.py`
   - `Env-rollout-parallel-segmentation*.py`
4. 对每个候选记录：ref、完整 commit SHA、脚本路径、最后修改提交和日期、命令行参数、输出目录、导入依赖、环境源码、metadata/config、`pyproject.toml` 与 `uv.lock` 的匹配关系。
5. 使用 `git show <commit>:<path>` 和 `git diff --name-status origin/main...<candidate>` 建立依赖闭包。不能只把单个入口脚本复制到当前 `main` 后直接宣称已恢复。
6. 排除或修复写向仓库外绝对路径的候选；任何输出路径都必须显式指向本仓库内。
7. 在隔离 worktree 或恢复分支中先运行 `uv run <script> --help`，再执行“单任务、单 episode、单 worker”的最小 smoke test。记录实际默认值，不能只相信 help 文本或注释。

当前初始化预检发现的线索：

- 当前 `main` 只有 `scripts/dataset_replay.py`，没有正式批量生成入口；
- 初步候选链位于 `origin/cvpr2026Challenge-heldOutSeed-4-5/4` 的 `2fa5660d8b78f31a6735538660d18a8e830bff63`，包括：
  - `scripts/dev3/Env-rollout-parallel-segmentation.py`
  - `scripts/dev3/Env-rollout-parallel-segmentationV2-withReplay.py`
  - `scripts/dev3/inspect_stat.py`
  - `scripts/dev3/env_specific_extraction/`
- 这只是 `/init` 期间的只读预检线索，尚未验证可用。候选脚本的部分 argparse 默认值与 help/docstring 存在不一致，因此必须完成依赖闭包检查和最小生成测试后才能选定。
- 更旧的显式生成器存在写向 `/data/hongzefu/data_0226/...` 等仓库外硬编码路径，不符合本任务约束，不能直接采用。

只有候选来源和依赖闭包明确、最小 smoke test 成功、产物确实位于仓库内，并且恢复/兼容性修改有清晰 diff 时，第二阶段才能标记为“完成”。

## 第三阶段：重新生成并做一致性审查

1. 保持官方参考数据只读。生成输出写入 `artifacts/generated/<candidate-commit>/`，日志与报告写入仓库内的独立目录。
2. 固定并记录 commit、依赖锁、随机种子、task、episode、difficulty、action space、worker 数、渲染设备和所有非默认参数。
3. 先完成单任务、单 episode 的小样本生成和审查；通过后再逐步扩大到全量。不得在小样本失败时直接启动全量生成。
4. 一致性审查至少覆盖：
   - 文件级：任务文件集合、文件数量、episode 数、大小和可用 checksum；
   - HDF5 结构级：group/dataset/attribute 名称、层级、dtype、shape、长度和缺失字段；
   - 元数据级：task ID、episode/seed、difficulty、task goal、action space、相机和环境配置；
   - 内容级：action、observation、状态、完成标志、轨迹长度和关键数值；严格相等与容差比较必须分别报告；
   - 行为级：用 `scripts/dataset_replay.py` 回放，并记录成功率、终止状态、异常和视频证据；
   - 测试级：先运行 `uv run python -m pytest tests/lightweight/`，再按环境条件运行 `uv run python -m pytest tests/dataset/`。
5. 必须区分“字节级一致”“结构一致”“数值容差内一致”和“行为一致”，不能用一次成功回放替代全量一致性结论。
6. 每个发现的差异都要记录参考值、生成值、影响范围、可能原因、是否阻塞以及后续动作。详细报告可以单独保存，但本文件必须保留摘要和报告路径。

只有目标范围内的数据全部生成、审查命令可复现、所有差异都有结论，并且汇总结果写入本文件后，第三阶段才能标记为“完成”。

## 当前进度

| 阶段 | 状态 | 已有证据 | 下一步 |
| --- | --- | --- | --- |
| `/init` 仓库初始化 | 完成 | 已确认根目录 `readme.md`、官方 dataset 链接、仓库内标准数据路径、当前 Git 状态及历史候选线索；已创建本文件 | 按第一阶段下载参考 dataset |
| 第一阶段：下载参考 dataset | 完成 | 固定官方 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`；16 个 HDF5、1,600 episode 的 SHA-256/HDF5 审计通过；16 任务双 GPU 回放共 160 episode、160 success 视频、无 worker 或 step 错误 | 可正式开始第二阶段：扫描 Git 历史并恢复最新可用生成脚本 |
| 第二阶段：恢复生成脚本 | 未开始，仅有预检线索 | 当前 `main` 无生成入口；已记录 `2fa5660...` 候选链，但未做依赖闭包或运行验证 | 等第一阶段完成后正式扫描和验证 |
| 第三阶段：生成与一致性审查 | 未开始 | 现有 replay/pytest 可作为部分验证手段，但尚无参考数据、恢复脚本或对比报告 | 等第二阶段完成后从单任务、单 episode 开始 |

## 追加式执行日志

### 2026-07-13 — `/init`

- 状态：完成。
- 操作：只读检查 `readme.md`、`tests/README.md`、`scripts/`、`.gitignore`、当前分支/工作树、远端 refs 和历史数据生成脚本路径；创建根目录 `AGENTS.md`。
- 关键证据：官方数据来源为 `Yinpei/robomme_data_h5`；标准本地读取路径为 `data/robomme_data_h5/`；README 中 Data Generation 段落已被注释且命令只是 `scripts/dev/xxxx` 占位符；当前 `main` 没有正式批量生成脚本。
- Git 状态：初始化检查时 `main` 位于 `6cea3594a7d2f475e124afa3c7575a24ac0b40ea`，跟踪 `origin/main`，修改本文件前工作树干净。
- 预检线索：记录了 `origin/cvpr2026Challenge-heldOutSeed-4-5/4` 的候选生成/回放/审查链，但没有把它判定为可用。
- 数据操作：未下载 dataset，未生成数据，未运行 Python。
- 下一步：执行第一阶段；开始前先在本节之后追加新日志，并同步更新“当前进度”表。

### 后续日志模板

复制下面的结构追加，不能删除已有日志：

```text
### YYYY-MM-DD HH:MM TZ — 第一/二/三阶段：<里程碑>

- 状态：进行中 / 受阻 / 完成。
- 目标：
- 执行命令：
- 输入与来源：
- 输出路径：
- 结果与证据：
- 差异或阻塞：
- 修改文件：
- 下一步：
```

### 2026-07-13 America/Detroit — 第一阶段：开始实施并行下载审计与回放

- 状态：进行中。
- 目标：在仓库内下载并审计官方 `Yinpei/robomme_data_h5`，并以 16 个 `spawn` 子进程完成按任务并行的回放 sanity check。
- 执行命令：预检已执行 `command -v uv`、`test -f pyproject.toml`、`test -f uv.lock`、`df -h .` 与 `nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader`；下载与回放命令待实现验证后执行。
- 输入与来源：README 官方链接 `https://huggingface.co/datasets/Yinpei/robomme_data_h5`。
- 输出路径：计划使用 `data/robomme_data_h5/`、`artifacts/reports/reference/<revision>/`、`runs/replay_videos/` 与仓库内 `.cache/`。
- 结果与证据：`uv` 位于 `/home/hongzefu/.local/bin/uv`；仓库有 `pyproject.toml`/`uv.lock`；`/data` 可用空间约 4.6 TB；GPU 0 和 GPU 1 均为 RTX 6000 Ada，空闲显存分别约 44,449 MiB、45,461 MiB。
- 差异或阻塞：尚未下载参考数据，未进行 HDF5 审计或环境回放；阶段不得标记为完成。
- 修改文件：`.gitignore`、`.dockerignore`、`AGENTS.md`，以及待新增的审计/回放实现和轻量测试。
- 下一步：实现审计脚本和 16 任务 `spawn` 并行回放，先完成轻量测试，再固定 Hugging Face revision 并下载。


### 2026-07-13 America/Detroit — 第一阶段：审计与并行回放实现已完成轻量验证

- 状态：进行中。
- 目标：完成参考数据审计入口与 16 任务同步 `spawn` 回放调度，随后开始固定版本下载。
- 执行命令：`uv run python -m py_compile scripts/dataset_replay.py scripts/audit_reference_dataset.py`（退出码 0）；`uv run --extra dev python -m pytest tests/lightweight/test_dataset_replay_parallel.py tests/lightweight/test_step_error_handling.py`（退出码 1）。
- 输入与来源：本仓库 `scripts/dataset_replay.py`、新建 `scripts/audit_reference_dataset.py` 与 `tests/lightweight/test_dataset_replay_parallel.py`。
- 输出路径：回放任务日志由 `--replay-log-dir` 指向 `artifacts/reports/reference/<revision>/replay_logs/`；汇总为同目录 `replay_summary.json`；视频保持在 `runs/replay_videos/joint_angle/`。
- 结果与证据：新增并行回放测试 2 项均通过，验证单任务 worker 在 barrier 释放后回传 mock 结果、源代码声明 `spawn`、16 任务和 GPU 0/1 分配；脚本语法编译通过。
- 差异或阻塞：既有 `tests/lightweight/test_step_error_handling.py::test_step_error_returns_status_error` 失败，原因是未修改的 `src/robomme/env_record_wrapper.py` 中 `DemonstrationWrapper.step()` 不含该测试预期的 `try/except`。此问题与参考数据下载及本次回放调度无关，未在第一阶段扩展修复。
- 修改文件：`.gitignore`、`.dockerignore`、`AGENTS.md`、`scripts/dataset_replay.py`、`scripts/audit_reference_dataset.py`、`tests/lightweight/test_dataset_replay_parallel.py`。
- 下一步：查询官方 dataset revision，使用仓库内缓存下载数据并运行 HDF5 审计。


### 2026-07-13 America/Detroit — 第一阶段：官方版本已固定，开始下载

- 状态：进行中。
- 目标：将固定版本官方参考数据完整下载到仓库内标准目录。
- 执行命令：`git ls-remote https://huggingface.co/datasets/Yinpei/robomme_data_h5 HEAD`（退出码 0）。
- 输入与来源：`Yinpei/robomme_data_h5`，完整 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。
- 输出路径：`/data/hongzefu/robomme_benchmark-restore-DataGen/data/robomme_data_h5/`；下载与 uv 缓存均位于仓库内 `.cache/`。
- 结果与证据：已获得 40 字符 HEAD SHA；尚未完成文件传输或 HDF5 审计。
- 差异或阻塞：无；下载中的网络、权限或磁盘错误将以实际退出码记录。
- 修改文件：`AGENTS.md`。
- 下一步：执行固定 revision 下载，随后审计 16 个 HDF5 文件和 1,600 个 episode。


### 2026-07-13 America/Detroit — 第一阶段：参考数据下载、解包与完整性审计通过

- 状态：进行中。
- 目标：确认官方参考数据完整后启动 16 任务、双 GPU 的并行回放 sanity check。
- 执行命令：`UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" uv run hf download Yinpei/robomme_data_h5 --repo-type dataset --revision a5e4e25ffe8af34f64944f9533d06455ce5f8337 --local-dir "$PWD/data/robomme_data_h5" --max-workers 8`（退出码 0）；`UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" uv run data/robomme_data_h5/tarxz_h5.py decompress --input_dir "$PWD/data/robomme_data_h5" --jobs 16`（退出码 0）；`UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" uv run scripts/audit_reference_dataset.py --h5-data-dir "$PWD/data/robomme_data_h5" --source-revision a5e4e25ffe8af34f64944f9533d06455ce5f8337 --report "$PWD/artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_audit.json"`（退出码 0）。
- 输入与来源：`Yinpei/robomme_data_h5`，固定 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`；官方下载的 16 个 `.h5.tar.xz` 由随附 `tarxz_h5.py` 在同一目录内解包，归档被保留。
- 输出路径：参考 HDF5 位于 `data/robomme_data_h5/`；审计报告位于 `artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_audit.json`。
- 结果与证据：审计 `passed=true`、HDF5 文件数 16、每任务 100 个 `episode_*`、总数 1,600、HDF5 总字节数 512,595,968,744、无符号链接、`errors=[]`；报告保存所有文件 SHA-256。
- 差异或阻塞：无完整性差异。参考目录同时保留官方 16 个 `.tar.xz` 归档和解包后的 `.h5`，目录占用约 530 GB；两者均被 Git/Docker 忽略。
- 修改文件：`AGENTS.md`、仓库内忽略的数据/报告/缓存产物。
- 下一步：运行同步 16 任务回放并记录每个任务日志、视频与汇总结果。


### 2026-07-13 America/Detroit — 第一阶段：官方参考 dataset 下载、审计与并行回放完成

- 状态：完成。
- 目标：下载固定版本官方参考数据，验证 16×100 demonstration，并完成 16 任务同步双 GPU 回放 sanity check。
- 执行命令：`git ls-remote https://huggingface.co/datasets/Yinpei/robomme_data_h5 HEAD`（退出码 0，得到 `a5e4e25ffe8af34f64944f9533d06455ce5f8337`）；根据 README 链接补充的 `UV_CACHE_DIR="$PWD/.cache/uv" HF_HOME="$PWD/.cache/huggingface" uv run hf download Yinpei/robomme_data_h5 --repo-type dataset --revision a5e4e25ffe8af34f64944f9533d06455ce5f8337 --local-dir "$PWD/data/robomme_data_h5" --max-workers 8`（退出码 0）；`uv run data/robomme_data_h5/tarxz_h5.py decompress --input_dir "$PWD/data/robomme_data_h5" --jobs 16`（退出码 0）；`uv run scripts/audit_reference_dataset.py --h5-data-dir "$PWD/data/robomme_data_h5" --source-revision a5e4e25ffe8af34f64944f9533d06455ce5f8337 --report "$PWD/artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_audit.json"`（退出码 0）；`set -o pipefail; uv run scripts/dataset_replay.py --h5-data-dir ./data/robomme_data_h5 --replay-log-dir artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/replay_logs 2>&1 | tee artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_replay_driver.log`（退出码 0）。
- 输入与来源：官方 dataset `Yinpei/robomme_data_h5`，完整 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`；官方发布为 16 个 `.h5.tar.xz`，使用其随附脚本在同一目录原地解包，归档保留。
- 输出路径：参考数据 `data/robomme_data_h5/`；完整审计报告 `artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_audit.json`；驱动日志 `artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/dataset_replay_driver.log`；每任务日志与汇总 `artifacts/reports/reference/a5e4e25ffe8af34f64944f9533d06455ce5f8337/replay_logs/`；视频 `runs/replay_videos/joint_angle/`。
- 结果与证据：审计 `passed=true`、HDF5 文件数 16、每任务 100 个 `episode_*`、总数 1,600、HDF5 总字节数 512,595,968,744、无符号链接、`errors=[]`；审计报告含每文件 SHA-256。同步 `spawn` 回放使用 GPU 0/1 各 8 个 worker，全部 16 个 task ready，进程退出码均为 0；汇总 `failures=[]`、`episodes_replayed=160`、`step_errors=0`、outcome 为 160 个 `success`，并产生 160 个 MP4 视频。
- 差异或阻塞：无参考数据完整性或回放差异。运行时仅出现 PyTorch、SAPIEN/URDF 的弃用/材质警告，未影响回放结果。既有 `tests/lightweight/test_step_error_handling.py::test_step_error_returns_status_error` 在未修改的 `DemonstrationWrapper.step()` 上失败，已单独记录，不阻塞第一阶段；新增 `tests/lightweight/test_dataset_replay_parallel.py` 2/2 通过，且 `py_compile` 退出码 0。
- 修改文件：`.gitignore`、`.dockerignore`、`AGENTS.md`、`scripts/dataset_replay.py`、`scripts/audit_reference_dataset.py`、`tests/lightweight/test_dataset_replay_parallel.py`；数据、归档、缓存、报告和视频均位于仓库内且被 Git/Docker 忽略。
- 下一步：正式开始第二阶段，扫描所有 branch、remote ref 与 tag，建立候选生成脚本及其依赖闭包，再在隔离 worktree 中完成最小 smoke test。
