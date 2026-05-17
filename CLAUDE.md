# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 协作准则

- **始终使用中文与用户沟通**：所有回复、解释、提交说明等面向用户的文本都用中文。
- **永远尽可能进行端到端测试**：完成代码改动后，优先用真实输入跑一遍完整流程验证效果，而不是仅依赖类型检查或单元测试；前端改动要在浏览器里实际操作一遍。如果环境不允许端到端测试，必须明确告知用户而不是假装通过。
- **永远使用 `uv run` 执行 Python**：运行脚本、测试、入口程序都通过 `uv run <command>` 调用，不要直接用 `python`、`python3` 或激活虚拟环境后再跑。安装/同步依赖也使用 `uv` 对应命令（如 `uv sync`、`uv add`）。

## 仓库概览

RoboMME 是一个面向「记忆增强机械臂操作」的基准（基于 ManiSkill）。当前分支 `VQA4challenge` 聚焦 `multi_choice` 动作空间（Video-QA 风格）的交互式运行与数据回放。`mani-skill` 通过 `pyproject.toml` 的 `[tool.uv.sources]` 钉在自有 fork（`YinpeiDai/ManiSkill` 指定 commit），不要换成上游版本。

## 常用命令

环境安装（首次）：

```bash
uv sync
uv pip install -e .
# 如果要跑 challenge_interface 的 server，再加一组：
uv sync --group server
```

运行示例：

```bash
# 单 episode 的 multi_choice 交互式 rollout（从 stdin 读选项；视频/日志写到 runs/sample_run_videos/）
uv run scripts/run_example.py --dataset test --task-id BinFill --episode-idx 0

# 批量评测（默认 joint_angle + DummyModel；遍历全部 16 个任务的 test 集）
uv run scripts/evaluation.py

# 回放 HDF5 数据集为视频（默认 multi_choice；写入 runs/replay_videos/）
uv run scripts/dataset_replay.py --h5-data-dir <your_h5_dir> --action-space-type multi_choice
```

测试（pytest 通过 `uv run` 调用，且根目录已注册 marker `slow / gpu / dataset / lightweight`）：

```bash
uv run python -m pytest tests/                              # 全部
uv run python -m pytest tests/lightweight                   # 不需要物理仿真，秒级
uv run python -m pytest tests/dataset                       # 需要 SAPIEN/ManiSkill，秒到分钟级
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors
uv run python -m pytest -m dataset                          # 用 marker 选
```

Challenge 本地联调（参赛者侧）：

```bash
# terminal 0
uv run python -m challenge_interface.scripts.deploy --port 8001
# terminal 1
uv run python -m challenge_interface.scripts.phase1_eval --port 8001
```

Docker：`docker build -t robomme:cuda12.8 .`，详见 `doc/docker_installation.md`。镜像启动时已经 `uv sync --group server`，容器内继续用 `uv run`。

## 架构要点

### 任务环境层 — `src/robomme/robomme_env/`
- 16 个任务每个一个 `.py`，通过 ManiSkill `@register_env` 注册，并由 `robomme_env/__init__.py` 统一 import。Gym 通过 `env_id`（如 `"VideoUnmask"`）拿到它们。
- `robomme_env/__init__.py` 末尾还会调用 `suppress_warnings()` 关闭 ManiSkill / gymnasium 的噪音 warning，避免淹没业务日志。
- 共享逻辑放在 `robomme_env/utils/`：planner、grounding 匹配（`oracle_action_matcher`、`choice_action_mapping`）、subgoal 评估、`task_goal` 文案生成、`vqa_options` 等。

### 包装器与入口 — `src/robomme/env_record_wrapper/`
唯一对外入口是 `BenchmarkEnvBuilder`（重新导出在 `env_record_wrapper/__init__.py`）。`make_env_for_episode(...)` 根据 `action_space` 选择不同的 wrapper 栈：

| `action_space` | 包装顺序（自内向外） |
| -------------- | -------------------- |
| `joint_angle`  | env → `DemonstrationWrapper` → `FailAwareWrapper` |
| `ee_pose`      | + `EndeffectorDemonstrationWrapper`（rpy 表征） |
| `waypoint`     | + `MultiStepDemonstrationWrapper`（关键帧步进） |
| `multi_choice` | + `OraclePlannerDemonstrationWrapper`（解析 `{"choice","point"}` 并调用 Panda motion planner） |

无论哪条路径都会被 `FailAwareWrapper` 兜底：底层 SAPIEN/IK/RRT 异常会被转成 `info["status"]="error"` 而不是抛出，调用方应当读 `info["status"]` 而不是写 `try/except`。

### Episode 元数据 — `src/robomme/env_metadata/{train,test,val}/`
每个 `(task, episode)` 的 `seed` 和 `difficulty` 写在 `record_dataset_<Task>_metadata.json`，由 `episode_config_resolver.load_episode_metadata` 读入并固定。基准评测必须复用这些值——构造 env 时不要自行覆盖 seed。

### 观测/动作约定
- 所有 obs 字段都是 **list**，因为部分任务先放 conditioning video 再执行；想拿当前帧用 `obs[key][-1]`。
- 可选字段通过 `make_env_for_episode(include_front_depth=..., include_*_extrinsic=..., include_available_multi_choices=..., ...)` 开关，默认全关；细节见 `doc/env_format.md`。
- `action_space="multi_choice"` 会**强制**打开 `front_camera_extrinsic_list` 和 `front_camera_intrinsic`（即使 `include_front_camera_*` 是 False），因为需要把 pixel `point` 反投到 3D。
- 坐标约定：`multi_choice` 的 `point` 是 **`[y, x]`**（行、列）在 front camera 图像上；`scripts/run_example.py` 从终端按 `<字母> <x> <y>` 读入后才转成 `[y, x]`。
- gripper：`-1`=闭合，`1`=张开，绝对动作。

### 数据格式
- HDF5 训练集结构（每个 `record_dataset_<Task>.h5`）：`episode_<N>/setup/`、`episode_<N>/timestep_<K>/{obs,action,info}`。完整字段表见 `doc/h5_data_format.md`。
- `dataset_replay.py` 用 `_build_action_sequence` 把回放序列按 action_space 反序列化（`waypoint` 会去除相邻重复，`multi_choice` 只取 `is_subgoal_boundary=True` 的步）。

### Challenge 接口 — `challenge_interface/`
参赛者只需要修改 `policy.py` 的 `Policy.infer(inputs) -> {"actions": ndarray}` 和 `Policy.reset()`，以及按需修改 `scripts/deploy.py`。`infer` 返回的 chunk 形状：`joint_angle` 是 `(chunk_size, 8)`，`ee_pose`/`waypoint` 是 `(chunk_size, 7)`。两种 transport（websocket / http）由 `deploy.py --transport` 选择。

### 测试缓存机制
`tests/dataset/` 下的用例通过 `tests/_shared/dataset_generation.py` 与 `dataset_factory` fixture 共享 motion planner 录制结果（按 `env_id / episode / difficulty / seed / mode_tag` 做 cache key）。同一组合的物理仿真在整个 pytest session 里只跑一次，不要在新测试里绕过这个 fixture 自己另起一次录制。
