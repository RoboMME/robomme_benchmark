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

## 数据生成管线 — `scripts/dev3/Env-rollout-parallel-segmentation.py`

数据生成主脚本是**薄实现**，所有 env-specific 行为必须委托给
`scripts/dev3/env_specific_extraction/{counting,imitation,permanence,reference}.py`
4 个 suite 模块。**这套契约是不可回退的**——任何回到主脚本里加 `if env_id == X` 分支
的改动都属于退化，必须在 4 个 suite 模块里加。

### 强制保持的目录与依赖隔离（**不可回退**）

```
scripts/dev3/
  Env-rollout-parallel-segmentation.py    # 薄编排：reset → 4 个 suite enrich → close
  env_specific_extraction/
    counting.py        # BinFill / PickXtimes / SwingXtimes / StopCube
    imitation.py       # MoveCube / InsertPeg / PatternLock / RouteStick + select_planner
    permanence.py      # ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap
    reference.py       # PickHighlight / VideoPlaceButton / VideoPlaceOrder / VideoRepick
    xy_common.py       # inspect 端共享
    counting_inspect.py / imitation_inspect.py / permanance_inspect.py / reference_inspect.py
```

**import 禁区**（grep 必须返回 0 行，否则 PR 不许合）：

- 主脚本 `Env-rollout-parallel-segmentation.py` 只允许 import 标准库、第三方包、
  `robomme.*` 包内模块、与 4 个 suite 模块（`counting / imitation / permanence /
  reference`）。**禁止** import `scripts/dev3/` 下除 `env_specific_extraction/` 之外
  的任何模块。
- 4 个 suite 模块自身 **禁止** import `scripts/dev3/` 下其它模块（`xy_common.py`
  只属于 inspect 端，rollout 端 4 个 suite 不要 import）。
- `scripts/dev/` 与 `scripts/dev2-snapshot-object/` 已整目录删除，**禁止重建**——
  历史脚本（`pickhighlight_setup_metadata.py` / `videorepick_setup_metadata.py` /
  `dataset-distribution.py` / `aggregate-visible-object-xy.py` 等）走 `git log`
  找回，不要再 import。

```bash
# 净化检查
grep -nE "^(from|import)\s+(pickhighlight|videorepick|counting|imitation|permanence|reference)" \
    scripts/dev3/Env-rollout-parallel-segmentation.py
# 期望：4 行 (counting / imitation / permanence / reference)

grep -nE "(from|import)\s+(scripts\.dev|pickhighlight|videorepick)" \
    scripts/dev3/env_specific_extraction/{counting,imitation,permanence,reference}.py
# 期望：0 行

test ! -d scripts/dev && test ! -d scripts/dev2-snapshot-object && echo "OK"
```

### 4 个 suite 模块的对外接口（对称契约）

每个 suite 模块都暴露**同名同签名**的 reset 钩子：

```python
def enrich_visible_payload(payload: dict, env: Any, env_id: str) -> None:
    """原地修改 visible_objects.json 的 payload —— 给本 suite 范围内的 env
    添加顶层字段。env_id 不属于本 suite 时是 no-op。"""
```

主脚本在 `_save_visible_objects_json` 里**必须按下面形态盲调 4 次**，不要折叠成
循环或单一 dispatch，也不要在主脚本里加任何 `if env_id == ...` 判断：

```python
counting.enrich_visible_payload(payload, env=env, env_id=env_id)
permanence.enrich_visible_payload(payload, env=env, env_id=env_id)
reference.enrich_visible_payload(payload, env=env, env_id=env_id)
imitation.enrich_visible_payload(payload, env=env, env_id=env_id)
```

每个 suite 内部第一行必须是 `if env_id not in <SUITE>_ENV_IDS: return`，确保
非自己范围内的 env 一律 no-op、零副作用。`No silent fallbacks`：自己范围内的 env
缺关键属性必须 raise，**不允许**静默返回。

execute 阶段 imitation 套件额外暴露：

```python
def select_planner(env, env_id) -> Optional[Any]:
    """PatternLock / RouteStick → FailAwarePandaStickMotionPlanningSolver
    (joint_vel_limits=0.3)；其他 env 返回 None，由主脚本实例化 default arm。"""
```

主脚本 `_create_planner` 形态固定为：

```python
suite_planner = imitation.select_planner(env, env_id)
if suite_planner is not None:
    return suite_planner
return FailAwarePandaArmMotionPlanningSolver(...)  # default arm
```

`FailAwarePandaStickMotionPlanningSolver` 不再在主脚本 import，由 imitation.py
延迟 import；default arm 仍属于主脚本（不是 imitation 套件特有逻辑）。

### `visible_objects.json` 是 reset 阶段唯一权威 sidecar

每个 episode 输出位置：
`runs/replay_videos/reset_segmentation_pngs/<env>_ep<n>_seed<s>/visible_objects.json`。

固定顶层字段（任何 env 都有）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `env_id` | str | 16 个 env id 之一 |
| `episode` | int | episode 索引 |
| `seed` | int | 当前种子 |
| `cameras` | list[str] | `["base_camera", "hand_camera"]` |
| `objects` | list[dict] | 通用可见对象列表（segmentation_id / name / object_type / world_xyz / visible_in） |

按 suite 范围**原地 enrich** 出来的顶层字段（由对应 suite 的
`enrich_visible_payload` 写入）：

| Suite | env 范围 | 字段 | 写入方 |
|---|---|---|---|
| Permanence | ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap | `permanence_init_state`（含 bins / cubes / swap_pairs / color_names / swap_times） | `permanence.enrich_visible_payload` |
| Reference (target) | PickHighlight / VideoPlaceButton / VideoPlaceOrder | `selected_target`（含 kind / task_target_indices/names/positions_xy/colors / all_candidates 等） | `reference.enrich_visible_payload` |
| Reference (VideoRepick) | VideoRepick | `videorepick_metadata`（含 target_cube_1_color + num_repeats） | `reference.enrich_visible_payload`（与 selected_target 解耦） |
| Counting / Imitation | 8 个 env | （无 enrich 字段） | no-op |

**禁止**再写任何独立 sidecar 文件（旧版的 `permanence_init_state.json` 已废弃）。
新增任何 reset 阶段 env-specific 数据必须走对应 suite 的
`enrich_visible_payload` 钩子写入 `visible_objects.json` 顶层。

### `runs/replay_videos/hdf5_files/` 不写任何 env-specific 字段

HDF5 setup group 只允许这 4 个**通用字段**：

```
episode_<N>/setup/{available_multi_choices, difficulty, seed, task_goal}
```

主脚本**禁止**调用任何"对 HDF5 setup group 追加 env-specific dataset"的逻辑
（旧版的 `pickhighlight_metadata` / `videorepick_metadata` dataset 已废弃）。
`_verify_setup_h5` 只验证通用结构 + `timestep_*` 不存在，**不做** env-specific
dispatch。

PickHighlight 的 target 颜色、VideoRepick 的颜色 + 重复次数等数据现在唯一权威源
是 `visible_objects.json` 顶层字段，下游 reader（`scripts/dataset_replay.py`、
`challenge_interface/` 等）需读这里。

### 改动 env-specific 行为的落点决策

新增 / 调整 env-specific 逻辑，永远先问"哪个 suite 的 env？"，再确定改动落点：

- **reset 阶段需要写新字段到 visible_objects.json**：在对应 suite 的
  `enrich_visible_payload` 加分支 + 在 docstring schema 表里登记字段名。
- **execute 阶段需要专属 planner / IK 配置**（目前只有 imitation 有）：在对应
  suite 模块加新公开接口（如 `select_planner`），主脚本调用一次即可。
- **跨 suite 共享的 reset 工具**（如 `_to_jsonable / _actor_xy / _actor_name`）：
  目前 permanence.py 与 reference.py 各自重复实现 —— 如果再有第三个 suite 用到，
  考虑抽到一个新文件 `env_specific_extraction/_rollout_common.py`（不要复用
  inspect 端的 `xy_common.py`，xy_common 只属于 inspect pipeline）。
- **永远不要**回到主脚本加 `if env_id in {...}` 分支；**永远不要**把 4 个 suite
  模块里的逻辑下沉回主脚本。
- **永远不要**让 4 个 suite 模块 import inspect 端模块（`xy_common.py` /
  `*_inspect.py`）；inspect 端可以反向 import 4 个 suite（用于发现/dedup）。

### 端到端验证规约（rollout 改动必跑）

任何对主脚本或 4 个 suite 模块的改动都必须跑一组覆盖全 16 env × 全难度的
端到端 setup-only rollout 后，再跑 inspect_stat 验证：

```bash
# 1) 16 env × 3 difficulties = 48 episodes
uv run python scripts/dev3/Env-rollout-parallel-segmentation.py \
    --env PickXtimes StopCube SwingXtimes BinFill \
          VideoUnmaskSwap VideoUnmask ButtonUnmaskSwap ButtonUnmask \
          VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight \
          InsertPeg MoveCube PatternLock RouteStick \
    --episode-number 3 --difficulty 1:1:1 \
    --gpu 0 --output-dir /tmp/refactor_e2e_rollout

# 2) 跑 inspect_stat
uv run python scripts/dev3/inspect_stat.py --base-dir /tmp/refactor_e2e_rollout
```

通过标准（缺一不可）：

- rollout 阶段 48/48 SUCCESS，0 retryable failure；
- HDF5 setup group 仅含 4 个通用字段（用 `h5py` 校验，无 `pickhighlight_metadata` /
  `videorepick_metadata`）；
- visible_objects.json 顶层按表中字段就位（permanence env 含 `permanence_init_state`、
  PickHighlight/VideoPlaceButton/VideoPlaceOrder 含 `selected_target`、VideoRepick
  含 `videorepick_metadata`、Counting/Imitation 顶层无 enrich 字段）；
- inspect_stat 输出 16 个 distribution PNG + 16 个 xy PNG，`parse_issues=0`、
  零 warning、`[Done] kept_hdf5=48 kept_json=48`。

---

## Dataset Inspection — `scripts/dev3/inspect_stat.py`

`inspect_stat.py` 把生成的 HDF5 + visible_objects.json 数据扫一遍，落地到
`runs/replay_videos/inspect-stat/` 下：

- `task-goal/episode_task_metadata.csv` + `<env>_distribution.png`（每个 env 的
  task-goal 分布）
- `xy/<env>_xy.png`（每个 env 的 visible-objects 散点图；4 个 permanence env 是
  3×N 的 3 行 PNG —— 行 1 visible-objects、行 2 cubes + swaps、行 3 first/second
  pickup bin；3 个 reference target env 含 selected_target overlay 的多面板 PNG）

### 强制保持的拆分结构（**不可回退合并**）

inspect_stat.py 的职责被严格限定在 **distribution pipeline + xy 编排**两件事。
xy 维度的所有 env-specific 渲染逻辑都已经按 4 个 task suite 拆到
`scripts/dev3/env_specific_extraction/` 下的独立模块，必须**坚持使用现在的拆分
数据结构**，禁止把 xy 渲染再合并回 inspect_stat.py。

```
scripts/dev3/
  inspect_stat.py                         # 仅做 distribution + 4 次薄调用编排
  env_specific_extraction/
    xy_common.py                          # 共享：dataclass / 发现 / 去重 / panel 绘制 / _render_xy_env
    counting_inspect.py                   # BinFill / PickXtimes / SwingXtimes / StopCube
    permanance_inspect.py                 # VideoUnmask / VideoUnmaskSwap / ButtonUnmask / ButtonUnmaskSwap
    reference_inspect.py                  # PickHighlight / VideoRepick / VideoPlaceButton / VideoPlaceOrder
    imitation_inspect.py                  # MoveCube / InsertPeg / PatternLock / RouteStick
    counting.py / imitation.py / permanence.py / reference.py
                                           # 4 个 suite 的 reset 钩子（rollout 端写入），同时为
                                           # 对应 *_inspect.py 提供 visible_objects.json 顶层字段
                                           # 的 discover/dedup 工具
```

inspect 端的 xy 渲染**只读 `visible_objects.json`** 一份文件——所有 env-specific
数据（`permanence_init_state` / `selected_target` / `videorepick_metadata`）都在
该文件顶层，不再存在独立 sidecar。distribution pipeline 也按这条原则：
PickHighlight 与 VideoRepick 的元数据从 `visible_objects.json` 顶层加载（通过
`segmentation_dir` 透传到 `_run_distribution_pipeline → _read_episode_rows →
_append_episode_row → _parse_semantic_fields`），而**不**读 HDF5 setup group。

每个 suite 模块对外只暴露一个 `visualize(...)` 函数，返回
`(kept_visible_files, skipped_visible_files)`。inspect_stat.py 在 `main()` 里**必须
按下面的形态分别调用 4 次薄调用**，不要折叠成循环或单一统一调用，也不要把任一
visualize() 的逻辑下沉回 inspect_stat：

```python
# Pipeline 1: pre-discover + dedup HDF5, then run distribution
kept_h5, skipped_h5, _, difficulty_map = _run_distribution_pipeline(
    hdf5_dir,
    task_goal_dir,
    args.env,
)

# Pipeline 2: xy 编排 — 4 个 suite 各自一次薄调用，渲染 visible_objects 散点图
kept_count, skipped_count = counting_inspect_module.visualize(
    segmentation_dir=segmentation_dir,
    output_dir=xy_dir,
    env_id=args.env,
    difficulty_by_env_episode=difficulty_map,
)

kept_perm, skipped_perm = permanance_inspect_module.visualize(
    segmentation_dir=segmentation_dir,
    output_dir=inspect_dir / "permanance_inspect",
    env_id=args.env,
)

kept_ref, skipped_ref = reference_inspect_module.visualize(
    segmentation_dir=segmentation_dir,
    output_dir=xy_dir,
    env_id=args.env,
    difficulty_by_env_episode=difficulty_map,
)

kept_imit, skipped_imit = imitation_inspect_module.visualize(
    segmentation_dir=segmentation_dir,
    output_dir=xy_dir,
    env_id=args.env,
    difficulty_by_env_episode=difficulty_map,
)
```

### 改动 xy 渲染时的落点决策

新增 / 调整渲染时，永远先问“哪个 suite 的 env？”，再确定改动落点：

- **某个 env 的 panel 组合 / 颜色 / 文案**：改对应的 `<suite>_inspect.py`
  （或 panel 行为本身在 `xy_common._panel_specs_for_env` / `_plot_panel`
  里按 env_id dispatch）。
- **跨 suite 共享的渲染原语 / dataclass / 去重 / 发现**：改 `xy_common.py`，
  保持 4 个 suite 模块都从这里 import。
- **suite 整体的 figure 布局**（例如 permanence 的 2 行布局）：改对应的
  `<suite>_inspect.py`，不要污染 xy_common。
- **distribution（task-goal CSV / 分布 PNG）**：改 inspect_stat.py 内部的 parsing /
  `_figure_specs_for_env` / `_render_env_figure`；这部分**不**要拆到 suite 模块里。
- **永远不要**让 suite 模块去 import inspect_stat，也不要再把 xy 渲染逻辑搬回
  inspect_stat.py。

### 端到端验证规约

任何对上述 5 个文件的改动都必须用真实数据集回归（默认是
`runs/replay_videos`）：

```bash
# 1) 备份基线
uv run python scripts/dev3/inspect_stat.py
cp -r runs/replay_videos/inspect-stat /tmp/inspect-stat.baseline

# 2) 应用改动 → 重跑
rm -rf runs/replay_videos/inspect-stat
uv run python scripts/dev3/inspect_stat.py

# 3) byte-diff（PNG + CSV 必须完全一致，除非本次改动确实意在改变像素）
diff -r --brief /tmp/inspect-stat.baseline runs/replay_videos/inspect-stat
```

PNG 比对必须用 `diff --brief` / `md5sum`，不要只看文件大小。matplotlib 在固定
backend (`Agg`) + 固定 dpi + 同一份输入下是 deterministic 的；任何 byte-level
diff 都说明数据流确实变了，不要忽略。

每个 suite 模块也支持独立 CLI：

```bash
uv run python scripts/dev3/env_specific_extraction/counting_inspect.py
uv run python scripts/dev3/env_specific_extraction/permanance_inspect.py
uv run python scripts/dev3/env_specific_extraction/reference_inspect.py
uv run python scripts/dev3/env_specific_extraction/imitation_inspect.py
```

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
