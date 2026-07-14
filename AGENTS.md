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
| 第二阶段：恢复生成脚本 | 完成 | 扫描 14 个远端 branch、0 tag、71 个关键词 commit 和 539 个历史路径；选定最新兼容 `a3842d1...`；最终唯一入口为 `scripts/generate_dataset.sh`，固化补丁为 `scripts/generate_dataset_a3842d1.patch`；候选 worktree/lock/Python 3.11.14、help、原 seed 1×1×1 smoke 与生成后契约均通过 | 已正式进入第三阶段 |
| 第三阶段：生成与一致性审查 | 完成（按用户修订的同 seed/离线完成口径） | 复用正式 16×9 共 144 条；metadata 原 seed attempt 1/1 均生成成功。新离线审计确认双方最终严格布尔完成率均为 144/144，`joint_strict_equal=false`，最大绝对差 `5.661269342205344e-9`；详细结论见 `scripts/reports/DATASET_COMPARISON_16x9.md` | 不得声称字节级、非 joint 全内容、数值容差或行为一致；若未来要求官方逐位值，需取得官方生成机数值运行时 |
| `dataset-gen` 分支整理与重新提交 | 完成 | 最终相对 `origin/main@6cea359...` 仅保留 `AGENTS.md` 与 5 个整理后的 `scripts/**` 文件；root lock/project、ignore 与 replay 均和 main 字节一致；加固后的 fresh BinFill 1×1×1、Python 编译、16×9 离线审计和路径白名单通过 | 本轮只创建本地 cleanup commit，不 push；默认入口支持 16×100，但本轮未实际运行全量 |

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


### 2026-07-13 23:56 EDT — 第二阶段：开始全历史扫描与生成链恢复

- 状态：进行中。
- 目标：更新并扫描全部 branch、remote ref 与 tag，建立数据生成候选的完整依赖闭包，选定最新可用版本后先完成单任务、单 episode、单 worker 的隔离 smoke test。
- 执行命令：`cat AGENTS.md`、`git status --short --branch`、`git remote -v`、`git rev-parse HEAD`、`git branch --show-current`、`command -v uv`、`ls -l pyproject.toml uv.lock`（均退出码 0）。
- 输入与来源：当前分支 `dataset-gen`，跟踪 `origin/dataset-gen`；开始 commit 为 `b41ab7f0b4cbebcb3cbb0a908827f4cafc60763d`；远端为 `https://github.com/RoboMME/robomme_benchmark`。
- 输出路径：历史扫描和候选报告计划写入 `artifacts/reports/recovery/`；隔离 worktree、缓存、生成数据和运行日志均只使用仓库内目录。
- 结果与证据：工作树开始时干净；`uv` 位于 `/home/hongzefu/.local/bin/uv`；根目录 `pyproject.toml` 与 `uv.lock` 均存在；第一阶段账本已证明可正式进入第二阶段。
- 差异或阻塞：尚未更新远端或判定任何候选可用；`2fa5660...` 仍只是预检线索。受限 shell 最初因 `bwrap` loopback 权限失败，随后获准以同样的只读命令完成预检，不影响仓库状态。
- 修改文件：`AGENTS.md`。
- 下一步：获取远端最新 refs，记录 branch/tag/SHA 快照，并行扫描关键词提交、历史路径和三个已知入口族。


### 2026-07-14 00:02 EDT — 第二阶段：远端快照完成并锁定官方生成链候选

- 状态：进行中。
- 目标：区分“最新历史入口”和“可重建官方完整 demonstration 的入口”，以官方 HDF5 实际 schema 与 metadata 作为候选筛选证据。
- 执行命令：`git fetch --all --tags --prune`、`git ls-remote --heads --tags origin`、`git for-each-ref ...`、`git log --all ...`、`git show <commit>:<path>`、`git diff --name-status origin/main...2fa5660...`（均退出码 0）；两条未正确引用 `--format=%(...)` 的首次 refs 格式化命令退出码 2，修正引用后退出码 0；官方 HDF5 检查使用 `UV_CACHE_DIR="$PWD/.cache/uv" uv run python -c ...`，其中两次错误假定根 group/字段名的探索命令退出码 1，修正为实际 `episode_N/timestep_N/{obs,action,info}` 后退出码 0。
- 输入与来源：远端 `origin` 当前 14 个 branch，无 tag；`origin/cvpr2026Challenge-heldOutSeed-4-5/4` 固定为 `2fa5660d8b78f31a6735538660d18a8e830bff63`；参考 HDF5 为第一阶段固定 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。
- 输出路径：本里程碑仅做只读扫描；后续恢复报告写入 `artifacts/reports/recovery/`，隔离 worktree 写入 `artifacts/recovery/worktrees/`。
- 结果与证据：`2fa5660.../scripts/dev3/Env-rollout-parallel-segmentation.py` 的 docstring 与校验逻辑明确是 setup-only，不执行 planner，故虽更新但不能重建官方完整 demonstration；其 `V2-withReplay` 能跑完整 rollout，但采用 heldout/dev3 seed 规则和后续环境语义。`fa05d07b0c71818625442ca270202ddf61df1e9e` 的提交标题为 `final v1 dataset train`，入口 `scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py` 严格读取 16 个 metadata JSON 中同 episode 的 seed/difficulty；官方 `BinFill/episode_0` 实测 seed=4000、difficulty=easy，且存在 `eef_state_raw`、`eef_action_raw` 和 `fail_recover_*`，与 `fa05d07...` 一致，而紧随其后的 `68a65a0...` 已移除这些字段。
- 差异或阻塞：`fa05d07...` 原入口把 metadata 与输出硬编码到 `/data/hongzefu/...`，不符合仓库内路径约束；必须以清晰补丁改成显式仓库内参数。尚未完成隔离 `--help` 和单 episode smoke test，因此还不能标记第二阶段完成。
- 修改文件：`AGENTS.md`。
- 下一步：记录 `fa05d07...` 的 pyproject/uv.lock 与完整源码依赖闭包，在仓库内隔离 worktree 应用最小路径兼容补丁，运行 `uv run ... --help` 和 1×1×1 smoke test。


### 2026-07-14 00:25 EDT — 第二阶段：隔离 smoke 成功但严格值验收未通过

- 状态：进行中。
- 目标：在不修改当前工作树源码的隔离 worktree 中验证 `fa05d07b0c71818625442ca270202ddf61df1e9e` 的真实 CLI、完整依赖和 1×1×1 生成能力，并在扩大生成前做官方数据逐叶子严格预比较。
- 执行命令：`git worktree add --detach artifacts/recovery/worktrees/fa05d07-original fa05d07b0c71818625442ca270202ddf61df1e9e`；`UV_CACHE_DIR="$PWD/.cache/uv" uv run scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py --help`（退出码 0）；应用仅含仓库内路径参数、固定 seed 单次尝试和缺失 episode 非零退出的临时补丁后，分别以 GPU 1、GPU 0、Python 3.11.13、2 workers/2 episodes 及历史 toppra wheel 运行 `uv run ... --env BinFill --episodes 1 --max-workers 1 ...` 或对应变体（均退出码 0）。所有 Python 命令前均确认 `command -v uv` 及候选 worktree 的 `pyproject.toml`/`uv.lock`。
- 输入与来源：候选 commit `fa05d07b0c71818625442ca270202ddf61df1e9e`；历史入口 `scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py`；metadata 来自同一 commit 的 `src/robomme/env_metadata/1206/`；官方参考为 `data/robomme_data_h5/record_dataset_BinFill.h5`。
- 输出路径：隔离 worktree `artifacts/recovery/worktrees/fa05d07-original/`；smoke 产物 `artifacts/generated/fa05d07b0c71818625442ca270202ddf61df1e9e/smoke*/`；日志 `artifacts/reports/recovery/fa05d07b0c71818625442ca270202ddf61df1e9e/smoke_BinFill_ep0.log`；历史 toppra wheel 备份 `artifacts/recovery/dependencies/fa05d07b0c71818625442ca270202ddf61df1e9e/`。
- 结果与证据：原始 `--help` 可运行，但确认 `--gpus` 的实际默认值是 `1`，help 文本错误写成 `0`；路径兼容补丁后的 `BinFill/episode_0` 使用官方 seed 4000、difficulty easy，完整生成 5 个子任务并成功合并 HDF5。与官方参考逐层比较时，18,160 个对象路径、schema、dtype、shape 和除 16 个浮点标量外的全部内容严格相等；仅 `timestep_7..22/action/joint_action` 的元素 4 存在绝对值约 `5e-20` 至 `2.2408256619612515e-18` 的末位差异。episode 1 同样仅有 15 个标量差异，最大绝对差 `1.1363794020363693e-17`。GPU 0/1、Python 3.11.13/3.11.14、1/2 workers 和两份 toppra 二进制均产生相同值，已排除这些变量。
- 差异或阻塞：小样本只达到结构一致和数值容差内一致，未达到严格值一致，因此不得声称“完全一致”，也不得按仓库规则直接扩大到 16×9。进一步发现旧 uv 环境使用 SciPy 1.17.1，而候选锁文件解析为 SciPy 1.17.0；该运行时漂移是下一项待验证变量。第三阶段仍保持“未开始”，本次预比较仅作为第二阶段候选验收。
- 修改文件：`AGENTS.md`、`scripts/compare_generated_dataset.py`、`tests/lightweight/test_compare_generated_dataset.py`；候选临时补丁、生成 HDF5、视频、wheel 和日志均在仓库内被忽略的 `artifacts/` 下，未覆盖官方参考数据。
- 下一步：用 uv 在隔离 worktree 验证 SciPy 1.17.1；若严格匹配则固化精确运行时，若仍不匹配则继续以历史环境二进制和数值调用链缩小来源。严格 smoke 通过后再完成第二阶段并启动 16×9。


### 2026-07-14 00:50 EDT — 第二阶段：最新兼容生成链恢复完成

- 状态：完成。
- 目标：完成全部 Git/ref 扫描、判定“最新且可用”的官方 train 生成链，固化完整依赖闭包与仓库内路径补丁，并证明恢复入口能用原 seed 生成官方结构数据。
- 执行命令：`git fetch --all --tags --prune`；关键词 commit/path 扫描；`git show <commit>:<path>`；`git diff --name-status origin/main...a3842d1...`；`git ls-tree a3842d1...`；`scripts/run_recovered_dataset_generator.sh --output-dir artifacts/generated/a3842d1.../runner-smoke-root --env BinFill --episodes 1 --max-workers 1 --gpus 1 --save-video`（退出码 0）；`uv run scripts/compare_generated_dataset.py ... --tasks BinFill --episodes 0 --rtol 1e-7 --atol 0 --allow-joint-action-allclose`（退出码 0）；比较器 8 项轻量测试退出码 0。
- 输入与来源：最新官方 HDF5/action 兼容 commit `a3842d1b77bc79e2f70cefcbab136207e7067065`，父提交 `6c9bbf9b8bde9127042b5d1850cf5f5fb60e7287`；入口最后修改为 `fa05d07b0c71818625442ca270202ddf61df1e9e`；metadata 固定为同一 tree 的 `src/robomme/env_metadata/train/`。
- 输出路径：tracked 恢复报告与补丁 `recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/`；运行器 `scripts/run_recovered_dataset_generator.sh`；smoke `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/runner-smoke-root/`；日志和比较报告 `artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/`。
- 结果与证据：扫描 14 个远端 branch、0 tag、71 个关键词 commit、539 个关键词历史路径。全历史较新的 `2fa5660.../dev3` 属 heldout seed/逐 episode 格式，不能生成官方 train；`68a65a...` 起删除官方 raw/fail-recovery 字段。`a3842d1...` 是删除发生前的最新兼容 commit。原入口硬编码 `1206/`，其中 4 条 seed 与官方不同；补丁改用已与官方 1,600 条 seed/difficulty 全量核对一致的 `train/`，并强制所有路径位于仓库内、原 seed 只尝试一次、缺失 episode 非零退出。固化运行器从 fixed commit 自动建 worktree、应用补丁并以候选 `uv.lock`/Python 3.11.14 启动。BinFill episode 0 使用 seed 4000 成功完成 5 个子任务并生成 HDF5/JSON/video。
- 差异或阻塞：完整叶子审查共 14,858 条 dataset record；仅 16 个 `action/joint_action[4]` float64 值不同，最大绝对差 `2.2408256619612515e-18`，`rtol=1e-7, atol=0` 全部通过。报告明确为 `strict_equal=false`、`accepted=true`、allowed=16、rejected=0。用户随后明确接受该微小差异，只要求同一原 seed 再次生成成功。旧 2 月本机产物与当前 smoke 在这些值上逐位相同，SciPy/GPU/Python/worker/toppra 变体均被排除；最可能但未确认的来源是不同 CPU/OpenBLAS kernel 的 SVD 末位差异。
- 修改文件：`AGENTS.md`、`recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/{README.md,generator-repo-local.patch}`、`scripts/run_recovered_dataset_generator.sh`、`scripts/compare_generated_dataset.py`、`tests/lightweight/test_compare_generated_dataset.py`。初次 a384 smoke 的 `tee` 因报告目录尚未创建而令整体退出码 1，但生成器实际成功；创建目录后在独立输出重跑退出码 0。固化运行器第一次测试发现相对 output-dir 会相对 worktree 解析，随后已修正为相对主仓库根目录，并以独立输出重跑退出码 0。
- 下一步：按用户批准的验收边界正式启动 16×9，并要求 144 条全部使用 metadata 原 seed 成功。


### 2026-07-14 00:50 EDT — 第三阶段：启动 16×9 固定 seed 生成与审查

- 状态：进行中。
- 目标：对全部 16 个 env 生成 episode 0–8（每 env 9 条，共 144 条），每条只允许 `train/` 中原 seed 的一次尝试；之后完成文件/schema/metadata/内容/行为/测试六层审查。
- 执行命令：计划执行 `scripts/run_recovered_dataset_generator.sh --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8 --episodes 9 --max-workers 9 --gpus 0,1 --save-video`；完整日志写入独立报告目录。
- 输入与来源：候选 commit `a3842d1b77bc79e2f70cefcbab136207e7067065`、候选 `uv.lock` SHA-256 `af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e`、同 tree 的 `train/` metadata、官方参考 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。
- 输出路径：生成数据 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`；报告 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`；官方 `data/robomme_data_h5/` 只读。
- 结果与证据：尚未启动正式长任务；smoke 已证明相同恢复链可生成并通过用户批准的容差策略。
- 差异或阻塞：验收策略只允许 `action/joint_action` 浮点叶子通过 `rtol=1e-7, atol=0`；schema、路径、dtype、shape、存储属性、metadata、离散值和其他内容仍须严格一致。任一 episode 原 seed 失败、任一非允许差异或任一 replay step error 都阻塞完成。
- 修改文件：`AGENTS.md`。
- 下一步：执行正式生成；生成器非零退出或缺少任一 episode 时停止，不启动后续扩大范围。


### 2026-07-14 01:18 EDT — 第三阶段：16×9 固定 seed 正式生成完成

- 状态：进行中。
- 目标：完成全部 16 个 env 的 episode 0–8，并证明每条都用 `train/` metadata 原 seed 一次生成成功；生成通过后才进入内容比较。
- 执行命令：`set -o pipefail; scripts/run_recovered_dataset_generator.sh --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8 --episodes 9 --max-workers 9 --gpus 0,1 --save-video 2>&1 | tee artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/generation_driver.log`（退出码 0）。启动前确认 `uv`/锁文件、输出目录不存在、磁盘剩余约 4.1 TB、GPU 0/1 空闲约 44.4/45.5 GiB。
- 输入与来源：恢复运行器固定 commit `a3842d1b77bc79e2f70cefcbab136207e7067065`、Python 3.11.14、候选 `uv.lock`、同 tree 的 `train/` metadata、`--max-seed-attempts 1`；workers=9、GPU=0,1、action space 沿候选生成器默认 `pd_joint_pos`、保存视频。
- 输出路径：数据 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`；完整驱动日志 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/generation_driver.log`。
- 结果与证据：生成器最终打印 `✓ All requested environments processed.` 并退出 0。日志精确包含 144 次 `attempt 1/1` 和 144 次 `[SUCCESS]`；无 episode 使用替代 seed。输出包含 16 个标准 `record_dataset_<Task>.h5`、16 个 metadata JSON、154 个生成视频，总占用约 48 GB，无文件系统符号链接。关键修复 seed 已实际成功：VideoPlaceButton episode 3 使用 10301，VideoPlaceOrder episode 2 使用 11206。每个任务都找到并合并 9 个 episode 文件。
- 差异或阻塞：部分 episode 的 screw planner 在同一 seed/同一 episode 内按历史逻辑尝试 3 次后 fallback 到 RRT* 并成功；这不是 seed 重试，不改变 setup seed。PatternLock/RouteStick 有非致命 URDF material 警告。生成视频数 154 大于 144，是部分任务一次 episode 产生额外视频文件；HDF5/metadata episode 数仍需由下一步结构审计确认。尚未运行全量内容比较，因此第三阶段不能完成。
- 修改文件：`AGENTS.md`；正式 HDF5、JSON、MP4 和日志均位于仓库内被忽略的 `artifacts/`，未改动只读参考数据。
- 下一步：先核对 16×9 HDF5/metadata/seed/difficulty 清单，再运行完整逐叶子比较；只有非 `joint_action` 差异为 0 且允许差异全部 allclose 时，才开始行为回放。


### 2026-07-14 01:31 EDT — 第三阶段：正式输出独立审计通过并硬化最终验收

- 状态：进行中。
- 目标：在长时间逐叶比较前排除旧输出、额外 episode、seed/metadata 漂移和过宽浮点容差，并固化可重复生成边界。
- 执行命令：只读扫描正式输出的 16 个 HDF5、16 个 metadata JSON、生成日志和视频；`uv run --extra dev python -m pytest tests/lightweight/test_compare_generated_dataset.py ...`（10 passed，退出码 0）；`bash -n scripts/run_recovered_dataset_generator.sh`（退出码 0）；分别以官方参考目录和正式非空目录作为 `--output-dir` 运行负向检查（均按预期退出 2）；最终比较命令为 `uv run scripts/compare_generated_dataset.py ... --episodes 0 1 2 3 4 5 6 7 8 --rtol 1e-7 --atol 0 --allow-joint-action-allclose --joint-action-max-abs-diff 1e-12`（正在运行）。每次 Python 命令前均确认 `command -v uv`、`pyproject.toml` 和 `uv.lock`。
- 输入与来源：正式输出 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`、候选 `train/` metadata、只读官方参考 HDF5。
- 输出路径：最终比较报告 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/comparison/`；旧规则下的中止部分报告保留为同级 `comparison-pre-hardening-partial/`，不得作为结论。
- 结果与证据：独立审计确认 16 个任务都严格只有 episode 0–8，共 144 条；candidate metadata、生成 JSON、生成 HDF5 setup 与官方参考的 seed/difficulty 144/144 一致，setup 所有字段的值、shape、dtype 144/144 精确一致；共 73,907 个连续 timestep，144/144 最终 `is_completed=True` 且 `simple_subgoal=All tasks completed`。正式日志包含 144 个 `attempt 1/1`、144 个 `[SUCCESS]`、无 Traceback/episode failure。16 个 HDF5 共 49,271,290,156 字节；144 个主视频覆盖完整，另有 10 个诊断视频。
- 差异或阻塞：代码审查发现运行器曾允许写入任意仓库内目录、比较器曾忽略生成侧额外 episode，且 joint_action 容差没有绝对误差上限。现已改为：输出只能位于候选专用目录的全新空子目录；拒绝官方目录和非空旧输出；worktree 完整 diff 必须与固化补丁逐字一致；生成侧额外 episode 一律阻塞；joint_action 除 `rtol=1e-7, atol=0` 外还必须满足最大绝对差 `<=1e-12`。第一次完整比较在旧代码下运行约 12 分钟后主动中止（退出码 130），未作为验收证据；最终规则比较已从头启动。5 次 screw planner 内部回退 RRT* 的 episode 最终均成功，不是 seed 重试。
- 修改文件：`AGENTS.md`、`scripts/run_recovered_dataset_generator.sh`、`scripts/compare_generated_dataset.py`、`tests/lightweight/test_compare_generated_dataset.py`；未修改官方参考数据或正式 HDF5。
- 下一步：等待最终逐叶比较完成；只有 `rejected_difference_count=0` 且所有允许差异绝对值不超过 `1e-12` 时，才启动 16×9 行为回放。


### 2026-07-14 01:52 EDT — 第二/三阶段：生成后轨迹契约与重复 seed 验证完成

- 状态：进行中（恢复器硬化完成，第三阶段完整比较仍在运行）。
- 目标：确保“生成器退出 0”确实对应完整 HDF5 轨迹落盘，并以同一原 seed 的独立重跑验证恢复链可重复使用。
- 执行命令：`uv run python -m py_compile scripts/validate_generated_dataset_contract.py scripts/compare_generated_dataset.py`（退出码 0）；两份定向轻测 `uv run --extra dev python -m pytest tests/lightweight/test_validate_generated_dataset_contract.py tests/lightweight/test_compare_generated_dataset.py ...`（24 passed，退出码 0）；正式 16×9 契约验证 `uv run --frozen scripts/validate_generated_dataset_contract.py ...`（退出码 0）；setup-only 负向样本契约验证（按预期退出码 1）；最终运行器 `--no-save-video` 负向检查（按预期退出码 2）；最终恢复链以 `--env BinFill --episodes 1 --max-workers 1 --gpus 1 --save-video` 在全新目录重跑（退出码 0）；两次有效 seed 4000 smoke 以 `rtol=0, atol=0` 严格比较（退出码 0）。所有 Python 命令前均确认 uv 与两份锁文件。
- 输入与来源：候选 `a3842d1...` 的 `train/` metadata；正式输出 `official-train-episodes-0-8/`；首次有效 `runner-smoke-root/`；最终有效 `runner-smoke-final-contract-v2/`。
- 输出路径：验证器 `scripts/validate_generated_dataset_contract.py`；正式契约报告 `artifacts/reports/generated/a3842d1.../official-train-episodes-0-8/generation_contract.json`；最终 smoke 日志 `artifacts/reports/recovery/a3842d1.../runner_smoke_final_contract_v2.log`；重复 seed 比较 `artifacts/reports/recovery/a3842d1.../repeated-seed-valid-smoke-comparison/`。
- 结果与证据：正式契约报告 `passed=true`，16 env、144 episodes、73,907 timesteps、0 errors；episode/生成 JSON/source metadata/HDF5 setup 的 seed 和 difficulty 类型及值一致；每条 timestep 连续，最终 `is_completed` 为严格 bool true，`simple_subgoal=All tasks completed`，无 symlink/temp 残留。最终 BinFill smoke 再次以原 seed 4000、attempt 1/1 完成全部 5 子任务，自动契约通过。两次有效 smoke 的 14,858 个 dataset 叶记录差异为 0，canonical SHA-256 同为 `13b2dbdf23a2cfae3b515d6d6be7d34704bf99fb88a4c4e2905f52c33ffa1504`。
- 差异或阻塞：探索时发现历史 `--no-save-video` 不只关闭 MP4，还会关闭 timestep 记录；该次进程虽打印 SUCCESS，但 HDF5 只有 setup，严格比较产生 18,152 个缺失对象，不能视为有效生成。已修复为运行器固定记录模式并拒绝该参数；契约验证器也新增至少一个连续 timestep 和最终完成标志校验，旧 setup-only 样本现按预期报 `h5_no_timesteps`。此探索产物仅作负向证据，不影响正式 16×9（正式命令使用 `--save-video` 且已有 73,907 timestep）。
- 修改文件：`scripts/validate_generated_dataset_contract.py`、`tests/lightweight/test_validate_generated_dataset_contract.py`、`scripts/run_recovered_dataset_generator.sh`、`recovery/a3842d1.../README.md`、`AGENTS.md`；未修改正式 HDF5 或官方参考数据。
- 下一步：等待最终完整逐叶比较结束；通过后按隔离视频目录执行 144 条行为回放。


### 2026-07-14 02:02 EDT — 第三阶段：完整内容比较结束并定位 PatternLock 单 episode 下游差异

- 状态：进行中；严格内容/原 joint-only 策略未通过，行为一致性正在验证。
- 目标：完整扫描 16 env × 9 episode 的全部 HDF5 对象、元数据和内容，明确区分严格一致、容差一致、下游状态/渲染差异与行为成功。
- 执行命令：最终命令 `uv run scripts/compare_generated_dataset.py --reference-dir data/robomme_data_h5 --generated-dir artifacts/generated/a3842d1.../official-train-episodes-0-8 --episodes 0 1 2 3 4 5 6 7 8 --rtol 1e-7 --atol 0 --allow-joint-action-allclose --joint-action-max-abs-diff 1e-12 ...`（退出码 1）；随后用 `jq -s` 对全部差异按 task/episode/path 聚合并写入 `difference_summary.json`。另用最终恢复运行器在全新目录重跑 PatternLock episode 0–1，再以 `rtol=0, atol=0` 与本次正式产物严格比较（生成与比较均退出码 0）。
- 输入与来源：官方参考 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`；正式生成候选 `a3842d1b77bc79e2f70cefcbab136207e7067065`；PatternLock 原 seed 15001/15100。
- 输出路径：完整报告 `artifacts/reports/generated/a3842d1.../official-train-episodes-0-8/comparison/`；差异聚合 `comparison/difference_summary.json`；PatternLock 重跑日志 `artifacts/reports/recovery/a3842d1.../runner_patternlock_repeat_episodes_0_1.log`；重跑严格报告 `artifacts/reports/recovery/a3842d1.../patternlock-formal-vs-repeat-strict/comparison.json`。
- 结果与证据：完整比较写出 1,996,551 条叶记录、5,963 条差异。16 个任务文件和生成侧 episode 集合均完整；对象层级、group/dataset/attribute、dtype、shape、存储属性、setup、seed、difficulty、task goal、相机/环境配置均未发现差异。全部 5,963 条均为 `dataset_content`；其中 5,355 条位于 `action/joint_action`。除 PatternLock episode 1 外，其余 episode 的非 joint 差异为 0。PatternLock episode 1 另有 608 条下游差异，涉及 eef action/state、joint state、camera extrinsic 和少量 RGB/depth 像素；数值最大绝对差 `4.76837158203125e-7`，图像层差异集中于 33 个 wrist RGB、16 个 wrist depth、5 个 front RGB 和 5 个 front depth 叶。所有 HDF5 对象仍存在且轨迹长度相同。
- 差异或阻塞：报告如实为 `strict_equal=false`、`passed=false`、`accepted=false`；原策略允许 5,271 条、拒绝 692 条。拒绝项由 608 条非 joint 下游差异及 84 条超过 `1e-12` 绝对上限/未通过 rtol 的 joint action 组成；PatternLock joint action 最大绝对差 `5.661269342205344e-9`，其余任务最大不超过 RouteStick 的 `1.1102230246251565e-15`。因此不能声称字节、严格内容或原 joint-only 策略一致。作为复现性取证，PatternLock episode 0/1 在当前恢复链再次用相同 seed attempt 1/1 成功，96/126 timestep 均最终完成；与本次正式产物的 6,006 个叶记录严格相等、差异 0。这证明当前恢复链可重复，官方差异来自官方历史运行与当前运行之间，而非本次同 seed 重跑漂移；后半句是基于证据的推断。是否按用户“同 seed 再次生成成功即可”的行为口径接受，仍需 144 条回放全部成功后定论。
- 修改文件：`AGENTS.md`；比较、聚合、PatternLock 重跑和报告均位于仓库内 `artifacts/`，未修改参考或正式 HDF5。
- 下一步：完成正在运行的 144 条 `joint_angle` 回放；严格检查 16 worker、144/144 outcome success、0 step error、144 个隔离视频，再运行完整 lightweight/dataset pytest。


### 2026-07-14 02:27 EDT — 第三阶段：16×9 行为回放、测试与最终验收完成

- 状态：完成（按用户修订的同 seed/行为验收口径）；严格内容比较未通过，相关限定继续保留。
- 目标：完成 16 个 env、每个 9 条原 seed 数据的行为回放、两套测试、文件 checksum 和差异定性，并把可复现结论写入仓库 Markdown。
- 执行命令：在独立 `behavior_replay/` 工作目录用 `uv run scripts/dataset_replay.py --h5-data-dir <official-train-episodes-0-8> --action-space-type joint_angle --replay-number 9 --replay-log-dir <behavior_replay/replay_logs>`（退出码 0）；`uv run --locked --extra dev python -m pytest tests/lightweight/`（退出码 1）；`uv run --locked --extra dev python -m pytest tests/dataset/`（退出码 0）；对 16 个生成 HDF5 和 16 个 metadata JSON 分别执行 `sha256sum`（均退出码 0）。所有 Python 命令前均确认 `command -v uv`、`pyproject.toml` 与 `uv.lock`。
- 输入与来源：正式生成 commit `a3842d1b77bc79e2f70cefcbab136207e7067065`、候选锁 SHA-256 `af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e`、候选 `train/` metadata 原 seed、只读官方 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`。
- 输出路径：最终结论 `recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/STAGE3_16x9_CONSISTENCY.md`；正式数据 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`；生成契约、逐叶比较、checksum、回放和测试日志均位于 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`。
- 结果与证据：正式生成和契约为 16 env、144 episode、73,907 timestep、144 次 attempt 1/1 成功、0 error。生成 HDF5 共 49,271,290,156 字节；HDF5/metadata SHA-256 清单各 16 行。行为回放 16 worker 全部 ready 且退出码 0，144/144 outcome `success`、0 step error、`failures=[]`；隔离目录有 144 个非空 MP4，共 125,986,178 字节。`tests/dataset/` 为 31 passed、369 warnings；`tests/lightweight/` 为 136 passed、3 failed、802 warnings。
- 差异或阻塞：逐叶比较退出码 1，报告保持 `strict_equal=false`、`accepted=false`：1,996,551 个叶中有 5,963 个内容差异，其中 5,355 个 joint-action 叶；原窄策略允许 5,271 条、拒绝 692 条。608 个非 joint 差异只出现在 PatternLock episode 1，涉及微小 EEF/state/extrinsic 与稀疏 RGB/depth 像素；该 episode 轨迹长度、完成标志和 replay 均成功。BinFill seed 4000 与 PatternLock seed 15001/15100 的当前环境独立重跑均 attempt 1/1 成功，且与本次当前产物逐叶严格相同。轻量测试的三项失败分别是未知环境空列表语义、SwingXtimes 连字符文案和已迁移到 `FailAwareWrapper` 的旧 AST 断言；相关测试/实现均早于本次恢复工作且本次未修改，不影响 16×9 生成契约或行为回放，但不得写成轻量测试全通过。
- 修改文件：新增 `recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/STAGE3_16x9_CONSISTENCY.md`，更新本 `AGENTS.md`；正式 HDF5、metadata、视频、日志、checksum 和 JSON/JSONL 报告均在仓库内被忽略的 `artifacts/`，官方参考保持只读。
- 下一步：第二、第三阶段在用户修订的行为/同 seed 范围内均已完成。若未来要求字节级或严格内容一致，现有证据表明需要取得官方生成机的 CPU/BLAS/仿真数值运行时；不能从本轮 144/144 成功回放反推严格内容一致。


### 2026-07-14 America/Detroit — `dataset-gen` cleanup：开始整理最终交付树

- 状态：进行中。
- 目标：以 `origin/main@6cea3594a7d2f475e124afa3c7575a24ac0b40ea` 为最终 tree 基线，保留已有两条提交并追加一个 cleanup commit；最终仅允许 `AGENTS.md` 与 `scripts/**` 不同于 main。
- 执行命令：已检查 `git status --short --branch`、`git rev-parse HEAD origin/main`、分支提交链、`git diff --name-status origin/main...HEAD`、根 `pyproject.toml`/`uv.lock` SHA-256，并在执行 Python 前确认 `command -v uv` 与两份 uv 管理文件。
- 输入与来源：当前 `dataset-gen@ecf9928ef4e29ae21099b3e496e7e2c4d860728f`；固定候选 `a3842d1b77bc79e2f70cefcbab136207e7067065`；现有正式生成数据 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/`。
- 输出路径：计划形成 `scripts/generate_dataset.sh`、`scripts/generate_dataset_a3842d1.patch`、`scripts/validate_generated_dataset_contract.py`、`scripts/compare_joint_actions.py`、`scripts/reports/DATASET_COMPARISON_16x9.md`；机器 JSON 仅写入仓库内 `artifacts/reports/`。
- 结果与证据：基线 SHA 已确认；根 `pyproject.toml` 与 `uv.lock` 当前均和 main 字节一致；本轮固定只复用 16×9，不补 episode 9、不运行 16×100，joint 差异只报告且不设通过阈值。
- 差异或阻塞：尚未完成脚本迁移、fresh 1×1×1 smoke、16×9 新报告、pytest 与最终路径级检查，因此 cleanup 不得标记为完成。
- 修改文件：`AGENTS.md`（本条开始日志）；后续只允许整理后的 `scripts/**`。
- 下一步：并行整理生成入口和 joint 审计工具；随后恢复 main 文件、删除旧路径并完成验证。


### 2026-07-14 America/Detroit — `dataset-gen` cleanup：代码归并与 16×9 离线 joint 报告完成

- 状态：进行中；脚本与报告已完成，fresh smoke、pytest 和提交尚未完成。
- 目标：将恢复链归并到 `scripts/`，并按最终离线口径重新审计现有 16×9 数据。
- 执行命令：`bash -n scripts/generate_dataset.sh`（退出码 0）；`scripts/generate_dataset.sh --help`（退出码 0，未要求 output-dir）；`sha256sum scripts/generate_dataset_a3842d1.patch`；`uv run --locked python -m py_compile scripts/compare_joint_actions.py scripts/validate_generated_dataset_contract.py`（退出码 0）；`uv run --locked scripts/compare_joint_actions.py`（修复 NumPy 元素索引 JSON 类型后最终退出码 0）。所有 Python 命令前均确认 `command -v uv` 和根 `pyproject.toml`/`uv.lock`。
- 输入与来源：只读官方 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`、正式生成候选 `a3842d1b77bc79e2f70cefcbab136207e7067065`、双方 episode 0–8；root lock SHA-256 `983de83f7b22c98b96c3c25a39958b4f5920e3232cfaa209c89542ef5639ac03`，candidate lock SHA-256 `af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e`。
- 输出路径：tracked 报告 `scripts/reports/DATASET_COMPARISON_16x9.md`；机器 JSON `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/joint_16x9/comparison.json`。
- 结果与证据：`validation_passed=true`、`joint_scope_complete=true`、`error_count=0`；16 个文件、144 条轨迹、73,907 个 joint 路径和 591,256 个 joint 元素完成核对；官方/生成最终严格布尔 `info/is_completed` 均为 144/144。joint 严格不相等：5,355 个路径、16,380 个元素有差异，最大绝对差 `5.661269342205344e-9`、mean abs `8.029681035134313e-13`、RMSE `5.061379303136063e-11`。差异只报告，无阈值、无“容差通过”结论。
- 差异或阻塞：首次 joint 扫描因最差元素索引保留 `numpy.int64` 而在 JSON 写出时报 TypeError、退出码 2；已最小修正为 Python `int` 并从头复跑成功。该失败未作为报告证据。新报告明确不声明字节级、非 joint 全内容、数值容差或行为回放一致。
- 修改文件：新增 `scripts/generate_dataset.sh`、`scripts/generate_dataset_a3842d1.patch`、`scripts/compare_joint_actions.py`、`scripts/reports/DATASET_COMPARISON_16x9.md`；最小扩展 `scripts/validate_generated_dataset_contract.py` 的预期 env/episode 闭环；恢复 main 的 ignore/replay 内容并撤出旧 `recovery/`、旧运行器、通用比较器、第一阶段审计脚本和分支新增测试。
- 下一步：在全新目录运行 BinFill 1×1×1 smoke，然后执行 main 自带 lightweight/dataset 测试与最终白名单检查。


### 2026-07-14 04:17 EDT — `dataset-gen` cleanup：最终 smoke、测试与提交前验收完成

- 状态：完成；本条与整理后的代码将由单个本地 cleanup commit 落盘，不推送远端。
- 目标：验证唯一生成入口在隔离候选环境中可重复生成完整轨迹，刷新正式 16×9 离线报告，并确保最终 tree 除 `AGENTS.md`、`scripts/**` 外与最新 main 完全一致。
- 执行命令：`bash -n scripts/generate_dataset.sh`、`scripts/generate_dataset.sh --help`（均退出码 0）；在故意注入 ambient `UV_*`、`GIT_*`、`PYTHONPATH` 与导出同名 `uv` shell function 的环境中运行 `scripts/generate_dataset.sh --output-dir artifacts/generated/a3842d1.../cleanup-smoke-final-v4 --env BinFill --episodes 1 --max-workers 1 --gpus 1`（退出码 0）；仓库外 output-dir 与非空 output-dir 负测均按预期退出 2；`uv run --locked python -m py_compile scripts/validate_generated_dataset_contract.py scripts/compare_joint_actions.py`（退出码 0）；`uv run --locked python scripts/compare_joint_actions.py`（退出码 0）；`uv run --locked python -m pytest tests/lightweight/`（退出码 1）；`uv run --locked python -m pytest tests/dataset/`（退出码 0）。每次 Python 命令前均确认 `command -v uv` 与根 `pyproject.toml`/`uv.lock`。
- 输入与来源：最新 main `6cea3594a7d2f475e124afa3c7575a24ac0b40ea`；生成候选 `a3842d1b77bc79e2f70cefcbab136207e7067065`；官方 revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`；正式 16×9 episode 0–8 数据。
- 输出路径：fresh smoke 为 `artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/cleanup-smoke-final-v4/`，契约为 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/cleanup-smoke-final-v4/generation_contract.json`；机器 joint JSON 为 `artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/official-train-episodes-0-8/joint_16x9/comparison.json`；tracked 报告为 `scripts/reports/DATASET_COMPARISON_16x9.md`；pytest 日志为 `artifacts/reports/cleanup/tests/`。
- 结果与证据：生成入口默认 16 env × 100 episode、20 workers、GPU 1，必须显式使用候选专用仓库内新目录；固定候选 lock、uv-managed Python 3.11.14、`train/` metadata、原 seed 单次尝试和记录模式。入口还拒绝 workspace mount/submount、路径 symlink、Git replace refs、非标准 index 标志、ambient Git/uv 配置及 shell function 劫持，并在 sync 后和生成后复核完整 worktree closure。最终 BinFill smoke 使用 seed 4000、attempt 1/1，550 个连续 timestep，最终严格布尔 `is_completed=true`，契约 `passed=true`/0 error；HDF5 SHA-256 为 `19e1ccf35f9bc3dcb254596ed008d2ee9b94e35b2b369aabc58a644a01b2239c`。固化补丁 SHA-256 为 `0336aa404ce805a160986857763ad89dbe72990d3afe662084a0d08d9c20c366`，与候选完整 worktree diff 逐字一致且只修改历史生成入口。
- 16×9 结论：`validation_passed=true`、`joint_scope_complete=true`、官方/生成离线完成率均为 144/144。73,907 个 joint 路径、591,256 个元素中，5,355 个路径和 16,380 个元素严格不等；差异元素比例 `2.7703735776042866e-2`，最大绝对差 `5.661269342205344e-9`、mean abs `8.029681035134313e-13`、RMSE `5.061379303136063e-11`。Joint 差异只报告、无容差阈值、不影响退出码 0；不据此声明字节、非 joint 全内容、容差或行为一致。
- 测试结论：`tests/dataset/` 为 31 passed、370 warnings；`tests/lightweight/` 为 109 passed、4 failed、803 warnings。四项失败均来自恢复为 main 的既有实现/测试期望：未知 env 返回 `[]` 而测试要求 `[""]`、SwingXtimes 文案含 `back-and-forth` 连字符、`DemonstrationWrapper.step()` 没有旧测试要求的 `try/except`、main 的 `dataset_replay.py` 没有旧测试要求的 status 检查；本次按范围约束未修改 main 源码或测试。可选 ruff 检查因 unchanged lock 未提供 ruff 而退出 2，未安装依赖、未修改 lock。
- 路径与版本验收：提交前白名单严格只有 `AGENTS.md`、`scripts/compare_joint_actions.py`、`scripts/generate_dataset.sh`、`scripts/generate_dataset_a3842d1.patch`、`scripts/reports/DATASET_COMPARISON_16x9.md`、`scripts/validate_generated_dataset_contract.py`；`pyproject.toml`、`uv.lock`、`.gitignore`、`.dockerignore` 与 `scripts/dataset_replay.py` 均和 main 字节一致，`git diff --check` 通过，Git 未跟踪 `data/`、`artifacts/`、视频或缓存。约 500 GB 本地数据仅由 `.git/info/exclude` 保护；该规则不提交且不保护 Docker context，因此当前含数据 checkout 不得直接作为 Docker build context。
- 修改文件：最终只保留上述 6 个白名单路径；删除旧 `recovery/`、旧运行器、通用全叶比较器、第一阶段审计脚本和分支新增的 3 个 lightweight 测试；`.gitignore`、`.dockerignore` 与 `scripts/dataset_replay.py` 精确恢复 main。
- 下一步：显式暂存白名单及必要删除，复核 cached diff/文件大小后创建本地 `Consolidate dataset generation and 16x9 joint audit` 提交；不 push，不补 episode 9，不执行 16×100。
