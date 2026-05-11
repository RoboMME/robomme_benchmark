# CLAUDE.md — RoboMME Benchmark

## 与用户沟通规范

- **始终使用中文**与用户沟通，包括所有文字回复、计划说明、错误提示和进度汇报。
- **计划（Plan）必须用中文撰写**，包括 Context、Approach、Verification 等各节。
- **代码、命令、变量名、注释**保持英文原样，不翻译。
- **端到端验证优先**：完成任何代码改动后，必须通过实际运行脚本或测试来验证功能正确性，除非用户明确指示跳过或说明无法运行（如缺少 GPU/display）。轻量单元测试只能作为辅助，不能代替端到端验证。

---

## Project Overview

RoboMME is a robotic benchmark for memory-augmented manipulation (CVPR 2026 challenge). It wraps [ManiSkill](https://github.com/haosulab/ManiSkill) environments into 16 tasks across 4 task suites. Participants implement a `Policy` class, serve it via `challenge_interface/`, and the organizer evaluates it end-to-end.

Source code lives in `src/robomme/`. Tests live in `tests/`. Challenge integration code lives in `challenge_interface/`.

---

## Environment & Package Management (uv)

This repo uses **uv** exclusively. Never use bare `pip`, `python`, or `conda` unless explicitly asked.

```bash
# First-time setup
uv sync
uv pip install -e .

# With optional server dependencies (challenge interface)
uv sync --group server

# Add / remove a dependency
uv add <package>
uv remove <package>

# Always run commands through uv
uv run scripts/run_example.py
uv run python -m pytest tests/
```

Python version is pinned at **3.11** (`.python-version`). `pyproject.toml` is the single source of truth; `uv.lock` must stay consistent with it.

`mani-skill` is sourced from a specific Git commit — do not change `[tool.uv.sources]` without a reason.

### Dependency Groups

| Group | Purpose | Install command |
|-------|---------|-----------------|
| (default) | core sim + training | `uv sync` |
| `dev` | pytest + opencv dev tools | `uv sync --group dev` |
| `server` | Flask + msgpack + websockets (challenge interface) | `uv sync --group server` |

---

## Error Handling Rules

**No silent fallbacks.** Every error path must raise explicitly. No `except Exception: pass`, no quiet defaults when something is missing, no returning `None` where a value is expected.

The **one sanctioned soft-error pattern** is `DemonstrationWrapper.step()`, which catches physics/IK exceptions and surfaces them as `info["status"] = "error"` + `info["error_message"]`. This is an explicit, structured error return — callers must check it.

- Script-level step loops must check `info.get("status") == "error"` and handle it explicitly (see `scripts/run_example.py`).
- Do **not** wrap `env.step()` in a bare `try/except Exception`.
- Do **not** add silent fallback branches or default-return paths unless the user explicitly requests graceful degradation.

---

## Running Tests

All test commands go through `uv run`.

```bash
# Full suite
uv run python -m pytest tests/

# Fast logic-only tests (no GPU, no physics engine)
uv run python -m pytest tests/lightweight/

# Heavy e2e tests (requires display/GPU — uses real sim)
uv run python -m pytest tests/dataset/

# By pytest mark
uv run python -m pytest -m lightweight
uv run python -m pytest -m dataset
uv run python -m pytest -m gpu
uv run python -m pytest -m slow

# Single file or test
uv run python -m pytest tests/lightweight/test_TaskGoal.py
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors

# With live output
uv run python -m pytest tests/ -s
```

### Test Categories

| Directory | Marks | Cost | What it covers |
|-----------|-------|------|----------------|
| `tests/lightweight/` | `lightweight` | Seconds | Pure logic: label matching, planners, AST checks, mocks |
| `tests/dataset/` | `dataset`, `gpu` | Minutes | Full env loop: sim + wrapper + HDF5 record/replay |

`tests/dataset/` uses a **session-scoped hash cache** (`_shared/dataset_generation.py`) so identical env/seed/difficulty combinations only generate data once per pytest session.

---

## Testing Philosophy

**Prefer end-to-end, heavy tests.** Tests that spin up real ManiSkill environments, record HDF5 datasets, and replay them are strongly preferred over pure unit tests.

- Do **not** mock the physics engine or the wrapper stack — divergence from real behavior has caused silent bugs before.
- Use mocks only for import-level stubs when ManiSkill/SAPIEN is genuinely unavailable in the current environment.
- When adding new behaviour, write an end-to-end test that: (1) builds a real `BenchmarkEnvBuilder`, (2) records a trajectory, (3) replays it and asserts the full output contract.
- Fall back to lightweight/AST tests only when physics rendering is genuinely unavailable.
- Do not skip or `xfail` without a comment explaining why — prefer fixing the root cause.

### Adding New Tests

- **Pure logic** → `tests/lightweight/test_<topic>.py`, mark `@pytest.mark.lightweight`
- **Sim-dependent** → `tests/dataset/test_<topic>.py`, mark `@pytest.mark.dataset`
- Register any new dataset fixture in `tests/dataset/conftest.py` following the existing `dataset_factory` pattern.

---

## Repository Layout

```
src/robomme/
  robomme_env/          # 16 task environments (one file per task)
  robomme_env/utils/    # shared helpers: planners, object generation, scoring
  env_record_wrapper/   # RecordWrapper, DemonstrationWrapper, BenchmarkEnvBuilder
  env_metadata/         # per-task episode metadata (train/val/test JSON)
  logging_utils.py

scripts/
  run_example.py        # quick single-env smoke test
  evaluation.py         # full eval harness
  dataset_replay.py     # replay downloaded HDF5 data for sanity check

challenge_interface/    # CVPR 2026 challenge server/client stubs
  policy.py             # participants implement Policy.infer() / Policy.reset()
  server_http.py / server.py
  scripts/deploy.py / phase1_eval.py

tests/
  lightweight/          # fast unit tests, no physics engine
  dataset/              # heavy e2e tests, real Mujoco sim + HDF5 recording
  _shared/              # shared fixtures and dataset generation cache
  conftest.py           # session-level repo_root / src_root fixtures

doc/                    # supplementary documentation
  env_format.md         # observation/action format spec
  h5_data_format.md     # HDF5 dataset format
  docker_installation.md
```

---

## Instantiating an Environment

```python
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    env_id="PickXtimes",          # task name
    dataset="test",                # "train" | "val" | "test"
    action_space="joint_angle",    # "joint_angle" | "ee_pose" | "waypoint" | "multi_choice"
    gui_render=False,
)
env = builder.make_env_for_episode(episode_idx=0)
obs, info = env.reset()
task_goal = info["task_goal"][0]

obs, reward, terminated, truncated, info = env.step(action)
# info["status"] == "error" means the step raised internally — do not ignore it
```

---

## Task Suites

| Suite      | Focus             | Tasks                                                        |
|------------|-------------------|------------------------------------------------------------|
| Counting   | Temporal memory   | BinFill, PickXtimes, SwingXtimes, StopCube                 |
| Permanence | Spatial memory    | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap|
| Reference  | Object memory     | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder|
| Imitation  | Procedural memory | MoveCube, InsertPeg, PatternLock, RouteStick                |

Dataset splits: `train` (100 ep/task), `val` (50 ep/task, fixed seeds), `test` (50 ep/task, held-out).

---

## Dev3 Pipeline — Rollout + Inspect（按需 Read）

`scripts/dev3/` 下两条 pipeline —— 数据生成 rollout（`Env-rollout-parallel-segmentation.py`
+ `env_specific_extraction/{counting,permanence,reference,imitation}.py`）与数据巡检
inspect_stat（`inspect_stat.py` + `env_specific_extraction/{*_inspect.py, xy_common.py}`）
—— 的强制规约统一抽到 [`doc/dev3_rollout_inspect_pipeline.md`](doc/dev3_rollout_inspect_pipeline.md)。

**改动以下任一文件时必须先 Read 该 doc**：

- `scripts/dev3/Env-rollout-parallel-segmentation.py`
- `scripts/dev3/inspect_stat.py`
- `scripts/dev3/env_specific_extraction/{counting,permanence,reference,imitation}.py`
- `scripts/dev3/env_specific_extraction/{counting_inspect,permanance_inspect,reference_inspect,imitation_inspect}.py`
- `scripts/dev3/env_specific_extraction/xy_common.py`

doc 内含：4 个 suite 模块对外契约、import 禁区、`visible_objects.json` 顶层字段表、
HDF5 setup 字段约束、xy / distribution 渲染分工、改动落点决策矩阵、端到端 e2e 验证规约
（rollout × 受影响 env × 全难度 → inspect_stat byte-diff）。

**与下方 "Val Seed 模型 Evaluate Pipeline" 严格分离**：dev3 是 seed 挑选阶段（生成 + 巡检），
evaluate 是 seed 验证阶段（部署模型 + phase1_eval），两边不互相 import / 不下沉对方逻辑。


---

## Challenge Interface (CVPR 2026)

```bash
uv sync --group server

# Terminal 0: policy server
uv run python -m challenge_interface.scripts.deploy --port 8001

# Terminal 1: local evaluation client
uv run python -m challenge_interface.scripts.phase1_eval --port 8001
```

Participants implement `Policy.infer()` and `Policy.reset()` in `challenge_interface/policy.py`.

> 本章节是面向 challenge 参赛者的通用说明。**Claude 在本仓库内执行 evaluate / 部署模型类任务时，
> 必须使用下方 "Val Seed 模型 Evaluate Pipeline" 章节的约束流程，与上面的 seed 挑选 pipeline
> （数据生成管线 + Dataset Inspection）保持严格分离。**

---

## Val Seed 模型 Evaluate Pipeline — Claude 自动化执行规范

本节是 **Claude 在 `cvpr2026Challenge-heldOutSeed-4-5/4` 分支自动执行 challenge_interface 评测**
的规范。**与上面"数据生成管线"（rollout 阶段挑 seed）与"Dataset Inspection"（inspect_stat 阶段挑
seed）严格分离 —— 两条 pipeline 不互相 import、不互相下沉逻辑，混用即视为退化。**

读到此节即表示 Claude 接到了 "evaluate 模型 / 验证 val seed / 部署 modul / 跑一下
perceptual-framesamp" 这类任务。严格按本节约束执行。

### 强制约束（不可绕过）

- **默认 = 1 task × 1 seed = `VideoUnmaskSwap` ep `0`**。无论用户措辞多模糊，Claude 接到
  evaluate 类请求时**默认只跑这一条 episode**，绝不主动扩 scope。
- **扩到 16 task × 1 seed（冒烟）必须先 AskUserQuestion 显式确认**。不要根据上下文揣测用户想跑
  16 个，必须让用户回答 "你确认要 16 个 env 都跑 ep 0 吗？（默认只跑 VideoUnmaskSwap ep 0）"。
- **永远不跑 16 task × 50 ep（800 episodes）或 16 task × 100 ep（1600 episodes）**。用户硬约束。
  若用户主动要求全量：给出命令清单让他自己在 tmux/screen 跑，**Claude 不替执行**（见末尾段落）。
- **不要再改 `challenge_interface/scripts/phase1_eval.py` 的 dataset 硬编码**：L210 + L232 已固定
  为 `"val"`（本分支约定）。既不要恢复成 `"test"`，也不要加 `--dataset` CLI 参数。
- **GPU 单卡共享**：deploy + sim 同 GPU 0（模型 ~12 GB Orbax 权重 + ManiSkill Vulkan sim ~10 GB，
  46 GB 卡刚好）。若 OOM 回退 `SAPIEN_RENDER_DEVICE=cpu MUJOCO_GL=osmesa`，**不**主动切多卡。

### 固定参数（不可变）

| 项 | 值 |
|---|---|
| Server 仓库 | `/data/hongzefu/robomme_policy_learning` |
| Client 仓库 | 本仓库 |
| Checkpoint | `run/ckpt/perceptual-framesamp-modul/79999`（policy_learning 内相对路径） |
| Transport | `websocket` |
| Port | `8001` |
| Dataset split | `val`（`phase1_eval.py` 与 `phase1_eval_single.py` 都已固定） |
| GPU | `CUDA_VISIBLE_DEVICES=0` |
| Default env / episode | `VideoUnmaskSwap` / `0` |

### Step 1 — 启动 policy server（后台）

Orbax 权重 ~12 GB，加载需 30 s – 2 min。**以端口 8001 LISTEN 为就绪标志**（deploy 进程加载
JAX/CUDA 时 stdout buffered，日志可能长时间无新行，不要靠 grep 日志关键字判断 ready）：

```bash
cd /data/hongzefu/robomme_policy_learning
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m challenge_interface.scripts.deploy \
  --transport websocket --host 0.0.0.0 --port 8001 \
  --checkpoint-dir run/ckpt/perceptual-framesamp-modul/79999 \
  > /tmp/deploy_modul_79999.log 2>&1 &
echo $! > /tmp/deploy_modul_79999.pid

until ss -lntp 2>/dev/null | grep -q ":8001\b"; do sleep 3; done
```

### Step 2 — 跑 evaluate

**默认 1 × 1（无需用户确认，直接跑）**：

```bash
cd /data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed
CUDA_VISIBLE_DEVICES=0 uv run python -m challenge_interface.scripts.phase1_eval_single \
  --env VideoUnmaskSwap --episode 0 \
  --transport websocket --host localhost --port 8001 \
  --team_id single_modul_VideoUnmaskSwap_ep0
```

`phase1_eval_single.py` 是本流程专属 wrapper，复用 `phase1_eval.py:run_episode()` 跑单
env × 单 episode，输出到 `challenge_results/<team_id>/videos/`。**不要再修改 `phase1_eval.py`
去支持单 env 过滤** —— 那条路径属于 "16 × 1 冒烟"。

**扩到 16 × 1（必须先 AskUserQuestion，否则不要跑）**：

```bash
cd /data/hongzefu/robomme_benchmark_cvpr2026-heldoutSeed
CUDA_VISIBLE_DEVICES=0 uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket --host localhost --port 8001 \
  --num_episodes 1 --action_space joint_angle \
  --team_id smoke16_modul
```

### Step 3 — 关 server

```bash
kill $(cat /tmp/deploy_modul_79999.pid); sleep 3
pkill -9 -P $(cat /tmp/deploy_modul_79999.pid) 2>/dev/null
```

### 端到端验证标准

每次 evaluate 跑完检查：

- `challenge_results/<team_id>/videos/` 下有对应 mp4，命名形如
  `<env>_ep_<idx>_<outcome>_<task_goal>.mp4`。
- 16 × 1 跑还要看 `progress.json`（含 16 env × 1 ep）与 `metrics.json`（per_task 16 项 + overall）。
- `outcome` 字段允许集合：`{success, fail, failure, timeout}` —— **不能出现 `"error"`**。
- 任何 `info["status"] == "error"` 都要追根因（按 "No silent fallbacks" 原则），不能忽略。

### 用户索要全量 50 ep × 16 env —— 给清单不替执行

用户主动要求全量（800 episodes）时，给以下命令清单 + 提示用户自己在 tmux/screen 内跑
（预计 6–12 小时）：

```bash
# (1) 启 server（同 Step 1，team_id 后缀建议改 _full）
# (2) 全量 client
CUDA_VISIBLE_DEVICES=0 uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket --host localhost --port 8001 \
  --num_episodes 50 --action_space joint_angle \
  --team_id val_modul_79999_full
# (3) 关 server（同 Step 3）
```

`phase1_eval.py` 支持 `progress.json` 断点续跑（基于 `_config_fingerprint`），中途 Ctrl-C 重跑同
命令即可。**Claude 不替用户跑这条命令**。

---

## Headless / GPU Notes

Tasks require a display for Vulkan rendering. For headless machines:

```bash
export SAPIEN_RENDER_DEVICE=cpu
export MUJOCO_GL=osmesa
```

Or use the provided Docker image:

```bash
docker build -t robomme:cuda12.8 .
docker run --rm -it --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
  -v "$PWD/runs:/app/runs" \
  robomme:cuda12.8
```

Tests marked `gpu` will fail without a working display or the env vars above. See `doc/docker_installation.md` for more options.
