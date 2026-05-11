# Dev3 Pipeline — Rollout + Inspect（seed 挑选 / 分布监控规约）

本文是 `scripts/dev3/` 下数据生成（rollout）与数据巡检（inspect_stat）pipeline
的强制规约。**仅在改动以下文件时按需 Read**：

- `scripts/dev3/Env-rollout-parallel-segmentation.py`（主 rollout 脚本）
- `scripts/dev3/env_specific_extraction/{counting,permanence,reference,imitation}.py`（4 个 suite 的 reset 钩子）
- `scripts/dev3/env_specific_extraction/{counting,permanence,reference,imitation}_inspect.py`（4 个 suite 的 inspect 渲染）
- `scripts/dev3/env_specific_extraction/xy_common.py`（inspect 端共享渲染原语）
- `scripts/dev3/inspect_stat.py`（distribution + xy 编排）

主 `CLAUDE.md` 里只保留一条指针；这里是完整规约。**与 challenge_interface
evaluate pipeline 严格分离**——本文是 seed 挑选阶段，evaluate 是 seed 验证阶段，两者不要相互 import 或下沉逻辑。

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

任何对主脚本或 4 个 suite 模块的改动都必须跑一组覆盖**受改动影响的 env ×
全难度**的端到端 setup-only rollout 后，再跑 inspect_stat 验证。

**判断"受影响 env"的依据 = 改动落在哪个 suite 模块**：

| 改动落点 | 必须跑的 env |
|---|---|
| `counting.py` / `counting_inspect.py` | BinFill / PickXtimes / SwingXtimes / StopCube |
| `permanence.py` / `permanance_inspect.py` | ButtonUnmask / ButtonUnmaskSwap / VideoUnmask / VideoUnmaskSwap |
| `reference.py` / `reference_inspect.py` | PickHighlight / VideoPlaceButton / VideoPlaceOrder / VideoRepick（按改动范围可只跑子集，例如只改 VideoRepick 时只跑 VideoRepick） |
| `imitation.py` / `imitation_inspect.py` | MoveCube / InsertPeg / PatternLock / RouteStick |
| `Env-rollout-parallel-segmentation.py` 主脚本（非 suite-local 改动） | 全 16 env |
| `xy_common.py` / `inspect_stat.py` | 全 16 env |

最小命令模板（按上表挑 env 替换）：

```bash
# 例：只动 reference.py 中 VideoRepick 的 enrich 钩子 → 只跑 VideoRepick × 3
uv run python scripts/dev3/Env-rollout-parallel-segmentation.py \
    --env VideoRepick \
    --episode-number 3 --difficulty 1:1:1 \
    --gpu 0 --output-dir /tmp/refactor_e2e_rollout

# 例：动 reference 全套 4 env
uv run python scripts/dev3/Env-rollout-parallel-segmentation.py \
    --env VideoRepick VideoPlaceButton VideoPlaceOrder PickHighlight \
    --episode-number 3 --difficulty 1:1:1 \
    --gpu 0 --output-dir /tmp/refactor_e2e_rollout

# 改完跑 inspect_stat 在 e2e 数据上
uv run python scripts/dev3/inspect_stat.py --base-dir /tmp/refactor_e2e_rollout
```

通过标准（缺一不可）：

- rollout 阶段所选 env × 3 难度 = N episodes 全部 SUCCESS，0 retryable failure；
- HDF5 setup group 仅含 4 个通用字段（用 `h5py` 校验，无 `pickhighlight_metadata` /
  `videorepick_metadata`）；
- visible_objects.json 顶层按表中字段就位（permanence env 含 `permanence_init_state`、
  PickHighlight/VideoPlaceButton/VideoPlaceOrder 含 `selected_target`、VideoRepick
  含 `videorepick_metadata`、Counting/Imitation 顶层无 enrich 字段）；
- inspect_stat 输出与所跑 env 数对齐（每 env 一张 distribution PNG + 一张 xy
  PNG），`parse_issues=0`、零 warning、`[Done] kept_hdf5=N kept_json=N`。

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
