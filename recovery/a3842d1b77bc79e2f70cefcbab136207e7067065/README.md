# 第二阶段恢复报告：RoboMME 官方 train dataset 生成链

## 结论

已选择 `a3842d1b77bc79e2f70cefcbab136207e7067065` 作为 Git 历史中最新的
官方 HDF5/action 兼容源码基线。它的后继
`68a65a099cf6ebf3ad20afe1dc12fcb4a050033c` 已删除
`obs/eef_state_raw`、`action/eef_action_raw` 和 `setup/fail_recover_*` 的写入，
因此不能生成官方参考 schema。

没有一个 clean commit 能单独复现官方数据：兼容入口硬编码读取 `1206/`
metadata，但官方 1,600 条数据的 seed/difficulty 与同提交的 `train/` 完全匹配。
恢复补丁因此明确固定 `train/`，同时移除仓库外输出、禁用 seed 递增重试，并在
请求 episode 缺失时以非零状态失败。

## 远端与全历史扫描

- 扫描时间：2026-07-14 EDT。
- 执行 `git fetch --all --tags --prune`，退出码 0。
- 远端共有 14 个 branch、0 个 tag；完整 ref/SHA 快照保存在
  `artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/refs_snapshot.txt`。
- 对提交标题执行大小写不敏感关键词搜索，共命中 71 个唯一 commit。
- 对全部历史路径搜索 `dataset|generate|record|rollout|demonstration|inspect|replay`，
  共命中 539 个唯一历史路径。
- 已用 `git log --all --full-history` 追踪
  `generate-dataset-control-seed-readJson-advanceV3.py`、`generate_dataset.py` 和
  `Env-rollout-parallel-segmentation*.py` 的新增、修改与删除。

关键远端 tip：

| ref | commit |
| --- | --- |
| `origin/main` | `6cea3594a7d2f475e124afa3c7575a24ac0b40ea` |
| `origin/dataset-gen` | `b41ab7f0b4cbebcb3cbb0a908827f4cafc60763d` |
| `origin/v0.5-choice-XY-3D2D` | `03c5253b47cd16e49122f5487de74ac2f208cfd1` |
| `origin/cvpr2026Challenge-heldOutSeed-4-5/4` | `2fa5660d8b78f31a6735538660d18a8e830bff63` |

## 候选比较

| 候选 | 入口与最后修改 | CLI / 输出 / metadata | 结论 |
| --- | --- | --- | --- |
| `2fa5660d8b78f31a6735538660d18a8e830bff63` | `scripts/dev3/Env-rollout-parallel-segmentationV2-withReplay.py`；入口最后修改 `406cb16b8f25ef104cb68146ca1a4cd22d38bafe`，2026-05-12 | 默认 2 个 env、200 episodes、difficulty ratio `[1,2,2]`、GPU 0、20 workers，输出 `runs/replay_videos`；seed 为 `1_500_000 + env_code*100000 + episode*100 + attempt`，逐 episode HDF5 并立即 replay | 是全历史较新的 heldout 生成/回放链，但 seed、schema 布局和文件布局均不是官方 16×100 train 合约，排除 |
| `03c5253b47cd16e49122f5487de74ac2f208cfd1` | `scripts/dev/generate_dataset.py`，2026-03-06 | 默认仅 4 个 env，采用新 seed offset/ratio；继承 raw/fail 字段已禁用的 writer | 晚于官方兼容边界，排除 |
| `68a65a099cf6ebf3ad20afe1dc12fcb4a050033c` | advanceV3 仍存在，2026-03-01 12:16 EST | 首次禁用官方 raw/fail-recovery HDF5 字段 | schema 边界提交，排除；全 refs 未发现后续重新启用 |
| `a3842d1b77bc79e2f70cefcbab136207e7067065` | `scripts/dev/generate-dataset-control-seed-readJson-advanceV3.py`；入口最后修改 `fa05d07b0c71818625442ca270202ddf61df1e9e`，2026-03-01 11:16 EST | 原始默认：全部 16 env、100 episodes、save-video=true、20 workers、GPU `1`；help 曾错误写 GPU 0；metadata/output 硬编码仓库外，失败时最多递增 seed 20 次 | 最新保留官方 HDF5/action 语义的提交；应用最小恢复补丁后选用 |
| `6c9bbf9b8bde9127042b5d1850cf5f5fb60e7287` | 与 `fa05` 的 generator、RecordWrapper、DemonstrationWrapper、锁文件 blob 完全相同，2026-03-01 12:06:13 EST | 与 `fa05` 相同 | 更保守但比选定提交早 27 秒；保留作等价备选 |
| `fa05d07b0c71818625442ca270202ddf61df1e9e` | advanceV3，标题 `final v1 dataset train` | 与选定提交相同的入口和 writer | 首个明确 train 生成候选，但不是最新兼容 commit |

`a3842d1...` 相比 `6c9bbf...` 只从 `DemonstrationWrapper` 返回 observation 中移除
`end_effector_pose_raw`；`RobommeRecordWrapper` 仍独立从 robot TCP 写入官方
`obs/eef_state_raw`。该变化不修改 planner、joint action、环境状态或 HDF5 action
写入路径。后续 smoke 与官方 schema 也验证了这一点。

## metadata 取证

选定提交内两个 metadata tree 不同：

- `src/robomme/env_metadata/1206` tree：`8a334a6da5785e6de9ec39ca175674fc8b588599`
- `src/robomme/env_metadata/train` tree：`625303a003f510fb474134331dad17e0d0b50d61`

官方 HDF5 与 `train/` 的 1,600 个 seed 和 difficulty 全部相同；`1206/` 有 4 条
seed 不同：BinFill 31、VideoPlaceButton 3/86、VideoPlaceOrder 2。计划的 episode
0–8 已包含其中两条，所以不能保留原入口的 `1206/` 默认值。

## 依赖闭包

恢复不是只复制入口，而是在隔离 worktree 使用完整候选 tree：

| 对象 | Git blob/tree |
| --- | --- |
| `.python-version` | `2c0733315e415bfb5e5b353f9996ecd964d395b2` |
| `pyproject.toml` | `620dc2402acb521f3163abc213ed3ec540cd9659` |
| `uv.lock` | `ac6531eda4e7dfd66e14a463342d18ca49fd36e2` |
| advanceV3 入口 | `105585b91dbfa9977ef5168542c125e2b1314f38` |
| 完整 `src/robomme` | tree `4de3deea9fd9dc9719c8c75f993e4a04920b53e9` |

文件 SHA-256：

- `pyproject.toml`：`105c71eac181e8a9facf97eec448e10760494063f724e6c7b2b403f5ac6483a8`
- `uv.lock`：`af4a645421c486ca1b1f27f5e54e8043497434b4efc49d2cbbf5eaa1b79d532e`
- `generator-repo-local.patch`：`0336aa404ce805a160986857763ad89dbe72990d3afe662084a0d08d9c20c366`

入口直接依赖 `gymnasium`、`h5py`、`numpy`、`torch`、完整环境注册、
`RobommeRecordWrapper`、failsafe planner 和 metadata。候选锁定 ManiSkill commit
`07be6fbc66350ddca200abfb0a11b692f078f7fd`，并解析为 Python 3.11、
NumPy 1.26.4、SciPy 1.17.0、toppra 0.6.3、mplib 0.1.1、torch 2.9.1、
h5py 3.15.1、SAPIEN 3.0.2。运行器固定 Python 3.11.14，并把 uv cache、Python、
worktree、数据和日志都放在当前仓库内。

按要求执行了：

```bash
git show a3842d1b77bc79e2f70cefcbab136207e7067065:<path>
git diff --name-status origin/main...a3842d1b77bc79e2f70cefcbab136207e7067065
```

第二条命令为空，因为候选是 `origin/main` 的祖先；完整候选 tree 和后续变更则由
`git ls-tree`、逐文件 `git show`、历史 blob 比较以及报告目录中的
`dependency_diff_vs_origin_main.txt` 共同记录。

## 恢复修改

补丁：`generator-repo-local.patch`。

修改范围仅为：

1. 将 metadata 固定为同一候选 tree 的 `train/`；
2. 增加必需的 `--workspace-root`、`--output-dir` 与可选 `--metadata-root`；
3. 拒绝解析到仓库外的 metadata/output；
4. 把 seed 尝试固定为 1，禁止换 seed 后冒充原 episode；
5. 请求 episode 未全部生成时抛出错误并返回非零；
6. 修正 `--gpus` help 中的实际默认值。

`scripts/run_recovered_dataset_generator.sh` 会从固定 commit 创建隔离 worktree、
验证并应用该补丁、逐字核对完整 generator diff 与补丁 SHA-256、拒绝 staged/
untracked/补丁范围外修改，再由候选自身 `uv.lock` 启动生成器。输出只能是
`artifacts/generated/a3842d1.../` 下不存在或为空的独立子目录；官方参考目录、
已有非空目录和仓库外目录都会在生成前被拒绝。

历史参数 `--no-save-video` 实际还会关闭 `RobommeRecordWrapper` 的 timestep 记录，
得到只有 setup 的 HDF5，因此恢复运行器固定开启记录模式并拒绝该参数。生成器退出
0 后，运行器继续调用 `scripts/validate_generated_dataset_contract.py`，严格检查：

1. HDF5 episode 集合连续且与生成 metadata 完全相同；
2. seed/difficulty 与候选 `train/` metadata、HDF5 setup 类型和值相同；
3. 每条轨迹至少有 `timestep_0`，全部 timestep 连续且为普通 group；
4. 最终 `info/is_completed` 必须为 scalar bool `true`，且
   `simple_subgoal` 必须为 `All tasks completed`；
5. 不存在符号链接或 `temp_*` 残留。

## 可用性验证

可复现命令：

```bash
scripts/run_recovered_dataset_generator.sh \
  --output-dir artifacts/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/runner-smoke-root \
  --env BinFill --episodes 1 --max-workers 1 --gpus 1 --save-video
```

退出码 0。使用 `train/` 中原 seed 4000、difficulty easy，5 个子任务全部成功，
生成 HDF5、metadata JSON 和视频均位于仓库内。日志为：

`artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/runner_smoke_root_BinFill_ep0.log`

运行器加固后又在全新目录以相同 seed 4000、单 episode、单 worker 重跑，生成器与
自动轨迹契约验证均退出 0。新日志为
`artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/runner_smoke_final_contract_v2.log`，
契约报告为
`artifacts/reports/generated/a3842d1b77bc79e2f70cefcbab136207e7067065/runner-smoke-final-contract-v2/generation_contract.json`。
两次有效 smoke 的 14,858 个 dataset 叶记录严格相等、差异为 0，task canonical
SHA-256 均为 `13b2dbdf23a2cfae3b515d6d6be7d34704bf99fb88a4c4e2905f52c33ffa1504`。
这直接证明同一 metadata seed 可以再次成功生成；报告位于
`artifacts/reports/recovery/a3842d1b77bc79e2f70cefcbab136207e7067065/repeated-seed-valid-smoke-comparison/`。

逐叶子比较命令使用 `scripts/compare_generated_dataset.py`。官方与 smoke 的 18,160
个对象路径、14,858 个 dataset、3,302 个 group、dtype、shape、存储属性、离散值、
图像、状态、EEF 和 metadata 均一致；只有 16 个
`action/joint_action[4]` float64 值存在最大 `2.2408256619612515e-18` 的差异，
全部通过 `rtol=1e-7, atol=0`。报告明确保留 `strict_equal=false`，并单独记录
`accepted=true`，没有把容差一致误写为字节级一致。

旧机本地 2026-02-27/28 的 20-episode 生成产物与当前 smoke 在这些值上逐位相同；
差异只存在于官方发布数据。该动作进入仿真前转为 float32 后完全相同，官方与重建
的 observation、EEF 和后续轨迹没有分叉。最合理但无法确认的原因是官方生成机与
当前 AMD EPYC 9334 的 OpenBLAS CPU kernel/SVD 末位差异。
